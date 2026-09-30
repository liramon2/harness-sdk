import { describe, expect, it } from 'vitest'
import { UNSET, AgentSpec, Choice, Fixed, Inherit, Open, Option, Preset, resolveSpec, defaultBuilder } from '../spec.js'
import type { ResolveSpecAxes } from '../spec.js'
import { Agent } from '../../agent/agent.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { createMockTool } from '../../__fixtures__/tool-helpers.js'

const AXES: ResolveSpecAxes = {
  presets: {},
  defaultPreset: undefined,
  instructions: new Open(),
  tools: new Choice(['read', 'shell'], true),
  mcpServers: new Inherit(),
  model: new Inherit(),
  context: new Fixed('none'),
}

describe('Choice', () => {
  describe('normalized', () => {
    it('wraps strings and fills UNSET', () => {
      const sentinel = { tag: 'sentinel' }
      const c = new Choice([new Option('alias', sentinel, 'd'), new Option('bare'), 'raw'])
      const opts = c.normalized()
      expect(opts).toHaveLength(3)
      expect(opts[0]!.value).not.toBe(UNSET)
      expect(opts[0]!.value).toBe(sentinel)
      expect(opts[0]!.description).toBe('d')
      expect(opts[1]!.value).toBe('bare') // UNSET filled from name
      expect(opts[2]!.name).toBe('raw')
      expect(opts[2]!.value).toBe('raw')
    })

    it('does not overwrite an explicit null value', () => {
      // null is the TS equivalent of Python's None — a legitimate value, distinct from UNSET.
      expect(new Choice([new Option('off', null)]).normalized()[0]!.value).toBeNull()
    })
  })

  describe('toSchemaProperty', () => {
    it('produces single enum and array enum', () => {
      expect(new Choice(['a', 'b']).toSchemaProperty()).toEqual({ type: 'string', enum: ['a', 'b'] })
      const prop = new Choice(['a'], true).toSchemaProperty('desc')
      expect(prop['type']).toBe('array')
      expect(prop['description']).toBe('desc')
    })

    it('folds option descriptions', () => {
      const prop = new Choice([new Option('x', UNSET, 'info'), 'y']).toSchemaProperty()
      expect(prop['description']).toContain('- x: info')
      expect(prop['description']).not.toContain('y:')
    })
  })

  describe('valueFor', () => {
    it('returns mapped value for known option and passthrough for unknown', () => {
      const c = new Choice([new Option('alias', 'real')])
      expect(c.valueFor('alias')).toBe('real')
      expect(c.valueFor('missing')).toBe('missing')
    })
  })
})

describe('resolveSpec', () => {
  it('applies preset by default', () => {
    const axes: ResolveSpecAxes = {
      ...AXES,
      presets: { g: new Preset({ instructions: 'help', context: 'all', lastMessages: 5 }) },
      defaultPreset: 'g',
    }
    expect(resolveSpec({ task: 'x' }, axes)).toEqual(
      new AgentSpec({
        task: 'x',
        agentType: 'g',
        instructions: 'help',
        tools: ['read', 'shell'],
        context: 'all',
        lastMessages: 5,
      })
    )
  })

  it('throws on unknown agent_type', () => {
    const axes: ResolveSpecAxes = { ...AXES, presets: { g: new Preset() }, defaultPreset: 'g' }
    expect(() => resolveSpec({ task: 'x', agent_type: 'nope' }, axes)).toThrow(/Unknown agent_type/)
  })

  it('rejects prototype keys, null, and non-string agent_type values', () => {
    const axes: ResolveSpecAxes = { ...AXES, presets: { g: new Preset() }, defaultPreset: 'g' }
    for (const bad of ['constructor', 'toString', '__proto__', ['g']]) {
      expect(() => resolveSpec({ task: 'x', agent_type: bad }, axes)).toThrow(/Unknown agent_type/)
    }
    // null falls back to default preset instead of throwing
    expect(resolveSpec({ task: 'x', agent_type: null }, axes)).toEqual(
      new AgentSpec({ task: 'x', agentType: 'g', tools: ['read', 'shell'], context: 'none' })
    )
  })

  it('suppresses default preset when ad-hoc instructions provided', () => {
    const axes: ResolveSpecAxes = {
      ...AXES,
      presets: { g: new Preset({ instructions: 'default' }) },
      defaultPreset: 'g',
    }
    expect(resolveSpec({ task: 'x', instructions: 'ad-hoc' }, axes)).toEqual(
      new AgentSpec({ task: 'x', instructions: 'ad-hoc', tools: ['read', 'shell'], context: 'none' })
    )
  })

  it('clamps tools to allowed set and parses last_messages', () => {
    expect(resolveSpec({ task: 'x', tools: ['read', 'write'], last_messages: '3' }, AXES)).toEqual(
      new AgentSpec({ task: 'x', tools: ['read'], lastMessages: 3, context: 'none' })
    )
  })

  it('sets lastMessages to undefined for invalid input', () => {
    for (const bad of ['abc', '3.5', '', ' ', null, [], false]) {
      expect(resolveSpec({ task: 'x', last_messages: bad }, AXES).lastMessages).toBeUndefined()
    }
  })

  it('Fixed ignores model-supplied value', () => {
    const axes: ResolveSpecAxes = { ...AXES, instructions: new Fixed('pinned') }
    expect(resolveSpec({ task: 'x', instructions: 'override' }, axes)).toEqual(
      new AgentSpec({ task: 'x', instructions: 'pinned', tools: ['read', 'shell'], context: 'none' })
    )
  })

  it('Choice maps option value', () => {
    const sentinel = { tag: 'model-sentinel' }
    const axes: ResolveSpecAxes = { ...AXES, model: new Choice([new Option('smart', sentinel)]) }
    expect(resolveSpec({ task: 'x', model: 'smart' }, axes).model).toBe(sentinel)
  })

  it('clamps preset tools to Choice set', () => {
    const axes: ResolveSpecAxes = {
      ...AXES,
      presets: { worker: new Preset({ tools: ['read', 'write'] }) },
      defaultPreset: 'worker',
      tools: new Choice(['read', 'shell'], true),
    }
    expect(resolveSpec({ task: 'x' }, axes)).toEqual(
      new AgentSpec({ task: 'x', agentType: 'worker', tools: ['read'], context: 'none' })
    )
  })

  it('resolves Fixed tools list', () => {
    const axes: ResolveSpecAxes = { ...AXES, tools: new Fixed(['a', 'b']) }
    expect(resolveSpec({ task: 'x' }, axes)).toEqual(new AgentSpec({ task: 'x', tools: ['a', 'b'], context: 'none' }))
  })

  it('resolves Fixed([]) to empty array, not undefined', () => {
    const axes: ResolveSpecAxes = { ...AXES, mcpServers: new Fixed([]) }
    expect(resolveSpec({ task: 'x' }, axes)).toEqual(
      new AgentSpec({ task: 'x', tools: ['read', 'shell'], mcpServers: [], context: 'none' })
    )
  })
})

describe('Fixed', () => {
  it('is frozen', () => {
    const f = new Fixed('x')
    expect(() => {
      ;(f as { value: unknown }).value = 'y'
    }).toThrow()
  })
})

describe('defaultBuilder', () => {
  const model = new MockMessageModel()

  it('builds a child Agent that inherits the parent model and forwards spec.name', () => {
    const parent = new Agent({ model, printer: false })
    const child = defaultBuilder(parent)(new AgentSpec({ task: 'x', name: 'worker' }))
    expect(child).toBeInstanceOf(Agent)
    expect(child.model).toBe(model)
    expect(child.name).toBe('worker')
  })

  it('inherits all parent tools when spec.tools is undefined', () => {
    const readTool = createMockTool('read', () => 'ok')
    const shellTool = createMockTool('shell', () => 'ok')
    const parent = new Agent({ model, tools: [readTool, shellTool], printer: false })

    const childToolNames = defaultBuilder(parent)(new AgentSpec({ task: 'x' }))
      .toolRegistry.list()
      .map((tool) => tool.name)

    expect(childToolNames).toEqual(['read', 'shell'])
  })

  it('resolves only matching tools and silently skips unknown names', () => {
    const readTool = createMockTool('read', () => 'ok')
    const shellTool = createMockTool('shell', () => 'ok')
    const parent = new Agent({ model, tools: [readTool, shellTool], printer: false })

    const childToolNames = defaultBuilder(parent)(new AgentSpec({ task: 'x', tools: ['read', 'nonexistent'] }))
      .toolRegistry.list()
      .map((tool) => tool.name)

    expect(childToolNames).toEqual(['read'])
  })
})
