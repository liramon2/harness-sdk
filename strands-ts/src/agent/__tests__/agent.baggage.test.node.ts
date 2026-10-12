import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { context, propagation, trace } from '@opentelemetry/api'
import { AsyncLocalStorageContextManager } from '@opentelemetry/context-async-hooks'
import { BasicTracerProvider, InMemorySpanExporter, SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base'
import { ALLOW_ALL_BAGGAGE_KEYS, BaggageSpanProcessor } from '@opentelemetry/baggage-span-processor'
import { Agent } from '../agent.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { createMockTool } from '../../__fixtures__/tool-helpers.js'
import type { Tool } from '../../tools/tool.js'
import { ToolResultBlock } from '../../types/messages.js'
import { AfterInvocationEvent } from '../../hooks/index.js'

const exporter = new InMemorySpanExporter()
const contextManager = new AsyncLocalStorageContextManager()

beforeAll(() => {
  context.setGlobalContextManager(contextManager.enable())
  trace.setGlobalTracerProvider(
    new BasicTracerProvider({
      spanProcessors: [new BaggageSpanProcessor(ALLOW_ALL_BAGGAGE_KEYS), new SimpleSpanProcessor(exporter)],
    })
  )
})

afterAll(() => {
  trace.disable()
  context.disable()
})

const tenant = (): string | undefined => propagation.getActiveBaggage()?.getEntry('tenant.id')?.value

// Tool that awaits `body` before returning an empty success result.
function asyncTool(name: string, body: () => Promise<void>): Tool {
  return createMockTool(name, (ctx) =>
    // eslint-disable-next-line require-yield
    (async function* () {
      await body()
      return new ToolResultBlock({ toolUseId: ctx.toolUse.toolUseId, status: 'success', content: [] })
    })()
  )
}

// Agent that calls `tool` once and then ends its turn.
function toolAgent(tool: Tool, baggageAttributes?: Record<string, string>): Agent {
  const model = new MockMessageModel()
    .addTurn({ type: 'toolUseBlock', name: tool.name, toolUseId: 't1', input: {} })
    .addTurn({ type: 'textBlock', text: 'done' })
  return new Agent({ model, tools: [tool], printer: false, ...(baggageAttributes && { baggageAttributes }) })
}

describe('Agent baggageAttributes', () => {
  it('defaults to an empty map and leaves the active baggage untouched', async () => {
    let seen: string | undefined = 'unset'
    const agent = toolAgent(createMockTool('probe', () => void (seen = tenant())))

    await agent.invoke('Hi')

    expect(agent.baggageAttributes).toStrictEqual({})
    expect(seen).toBeUndefined()
  })

  it('is current inside tools, merges with ambient baggage, and is stamped on the agent span', async () => {
    exporter.reset()
    let seen: Record<string, string | undefined> = {}
    const agent = toolAgent(
      createMockTool('probe', () => {
        const baggage = propagation.getActiveBaggage()
        seen = { tenant: baggage?.getEntry('tenant.id')?.value, ambient: baggage?.getEntry('ambient')?.value }
      }),
      { 'tenant.id': 'acme' }
    )
    const ambient = propagation.setBaggage(context.active(), propagation.createBaggage({ ambient: { value: 'up' } }))

    await context.with(ambient, () => agent.invoke('Hi'))

    expect(seen).toStrictEqual({ tenant: 'acme', ambient: 'up' })
    const agentSpan = exporter.getFinishedSpans().find((span) => span.name.startsWith('invoke_agent'))
    expect(agentSpan?.attributes['tenant.id']).toBe('acme')
    expect(tenant()).toBeUndefined()
  })

  it('isolates baggage between interleaved concurrent agents', async () => {
    const seen: Record<string, (string | undefined)[]> = { a: [], b: [] }
    // Each tool samples baggage before and after yielding to the other agent.
    const sampler = (name: 'a' | 'b'): Tool =>
      asyncTool(`tool_${name}`, async () => {
        seen[name]!.push(tenant())
        await new Promise((resolve) => setTimeout(resolve, 5))
        seen[name]!.push(tenant())
      })

    await Promise.all([
      toolAgent(sampler('a'), { 'tenant.id': 'acme' }).invoke('Hi'),
      toolAgent(sampler('b'), { 'tenant.id': 'globex' }).invoke('Hi'),
    ])

    expect(seen).toStrictEqual({ a: ['acme', 'acme'], b: ['globex', 'globex'] })
  })

  it('lets a nested agent override baggage for its scope and restores the outer value after', async () => {
    const seen: (string | undefined)[] = []
    const inner = toolAgent(
      createMockTool('inner_probe', () => void seen.push(tenant())),
      { 'tenant.id': 'inner-corp' }
    )
    const outer = toolAgent(
      asyncTool('outer_probe', async () => {
        seen.push(tenant())
        await inner.invoke('Hi')
        seen.push(tenant())
      }),
      { 'tenant.id': 'outer-corp' }
    )

    await outer.invoke('Hi')

    expect(seen).toStrictEqual(['outer-corp', 'inner-corp', 'outer-corp'])
  })

  it.each([
    ['consumer throws', (stream: AsyncGenerator) => stream.throw(new Error('boom')).catch(() => {})],
    ['consumer breaks early', (stream: AsyncGenerator) => stream.return(undefined)],
  ])('keeps baggage current for invocation cleanup when the %s', async (_, stop) => {
    let seen: string | undefined
    const agent = toolAgent(
      createMockTool('probe', () => {}),
      { 'tenant.id': 'acme' }
    )
    agent.addHook(AfterInvocationEvent, () => void (seen = tenant()))

    // Stop inside the model call so the invocation's cleanup (and AfterInvocationEvent) runs.
    const stream = agent.stream('Hi')
    while ((await stream.next()).value?.type !== 'beforeModelCallEvent');
    await stop(stream)

    expect(seen).toBe('acme')
  })
})
