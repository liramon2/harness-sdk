/**
 * Human-in-the-loop interrupt system for agent workflows.
 *
 * Interrupt Flow:
 * 1. Hook or tool calls `event.interrupt()` or `context.interrupt()`
 * 2. If resuming (response exists), the response is returned
 * 3. Otherwise, agent execution halts with `stopReason: 'interrupt'`
 * 4. User resumes by invoking agent with `interruptResponse` content blocks
 * 5. On resume, `interrupt()` returns the user's response
 */

import { InterruptResponseContent, type InterruptResponseContentData, type InterruptParams } from './types/interrupt.js'
import { deepCopy, type JSONValue } from './types/json.js'
import type { LocalAgent } from './types/agent.js'
import { Message, ToolResultBlock, type MessageData, type ToolResultBlockData } from './types/messages.js'
import type { Snapshot } from './types/snapshot.js'
import type { Agent } from './agent/agent.js'
import { logger } from './logging/logger.js'

/**
 * Origin of an interrupt:
 * - `'tool'` — raised by a tool callback via `ToolContext.interrupt()`.
 * - `'hook'` — raised by an agent-level hook (e.g. `BeforeToolCallEvent.interrupt()`).
 * - `'multiagent-hook'` — raised by a multi-agent hook (e.g. `BeforeNodeCallEvent.interrupt()`).
 */
export type InterruptSource = 'tool' | 'hook' | 'middleware' | 'multiagent-hook'

/**
 * Represents an interrupt that can pause agent execution for human-in-the-loop workflows.
 */
export class Interrupt {
  /**
   * Unique identifier for this interrupt.
   */
  readonly id: string

  /**
   * User-defined name for the interrupt.
   */
  readonly name: string

  /**
   * User-provided reason for raising the interrupt.
   */
  readonly reason?: JSONValue

  /**
   * Human response provided when resuming the agent after an interrupt.
   */
  response?: JSONValue

  /**
   * Where this interrupt was raised from — a tool callback, an agent-level hook, or
   * a multi-agent orchestrator hook. Always populated. When deserializing a snapshot
   * produced by an older SDK that did not record this field, defaults to `'hook'`.
   */
  readonly source: InterruptSource

  constructor(data: { id: string; name: string; reason?: JSONValue; response?: JSONValue; source?: InterruptSource }) {
    this.id = data.id
    this.name = data.name
    if (data.reason !== undefined) {
      this.reason = data.reason
    }
    if (data.response !== undefined) {
      this.response = data.response
    }
    // Default for legacy snapshots that predate the `source` field; current code
    // paths always supply a value explicitly.
    this.source = data.source ?? 'hook'
  }

  /**
   * Serializes the interrupt to a JSON-compatible object.
   */
  toJSON(): { id: string; name: string; reason?: JSONValue; response?: JSONValue; source: InterruptSource } {
    return {
      id: this.id,
      name: this.name,
      ...(this.reason !== undefined && { reason: this.reason }),
      ...(this.response !== undefined && { response: this.response }),
      source: this.source,
    }
  }

  /**
   * Creates an Interrupt instance from a JSON object.
   *
   * @param data - JSON data to deserialize
   * @returns Interrupt instance
   */
  static fromJSON(data: {
    id: string
    name: string
    reason?: JSONValue
    response?: JSONValue
    source?: InterruptSource
  }): Interrupt {
    return new Interrupt(data)
  }
}

/**
 * Error thrown when human input is required to continue agent execution.
 * Caught by the agent loop to trigger an interrupt stop.
 */
export class InterruptError extends Error {
  /**
   * The interrupts that caused this error.
   */
  readonly interrupts: Interrupt[]

  constructor(interrupt: Interrupt | Interrupt[]) {
    const all = Array.isArray(interrupt) ? interrupt : [interrupt]
    const message =
      all.length === 1
        ? `Interrupt raised: ${all[0]!.name}`
        : `${all.length} interrupts raised: ${all.map((i) => i.name).join(', ')}`
    super(message)
    this.name = 'InterruptError'
    this.interrupts = all
  }
}

/**
 * Data format for serialized interrupt state.
 */
export interface InterruptStateData {
  /**
   * Map of interrupt IDs to interrupt data.
   */
  interrupts: Record<string, { id: string; name: string; reason?: JSONValue; response?: JSONValue }>

  /**
   * Resume responses that were provided when resuming from an interrupt.
   */
  resumeResponses?: InterruptResponseContentData[] | undefined

  /**
   * Whether the agent is in an interrupted state.
   */
  activated: boolean

  /**
   * Pending tool execution state for resume after interrupt.
   */
  pendingToolExecution?: PendingToolExecution | undefined

  /**
   * Additional state carried for the duration of an interrupt, e.g. an interrupted sub-agent turn.
   */
  context?: Record<string, JSONValue> | undefined
}

/**
 * Pending tool execution state stored when an interrupt occurs mid-execution.
 * Contains all data needed to resume tool execution without re-calling the model.
 */
export interface PendingToolExecution {
  /**
   * The assistant message containing tool use blocks, serialized as MessageData.
   */
  assistantMessageData: MessageData

  /**
   * Tool results that were completed before the interrupt.
   * Maps toolUseId to serialized ToolResultBlock data.
   */
  completedToolResults: Record<string, { toolResult: ToolResultBlockData }>
}

/**
 * Tracks the state of interrupt events raised during agent execution.
 *
 * Interrupt state is cleared after resuming.
 */
export class InterruptState implements InterruptStateData {
  /** Record of interrupt IDs to Interrupt instances. */
  interrupts: Record<string, Interrupt>

  /** Resume responses provided when resuming from an interrupt. */
  resumeResponses?: InterruptResponseContent[] | undefined

  /** Whether the agent is in an interrupted state. */
  activated: boolean

  /** Pending tool execution state for resume. */
  pendingToolExecution?: PendingToolExecution | undefined

  /** Additional state carried for the duration of an interrupt, e.g. an interrupted sub-agent turn. */
  context: Record<string, JSONValue>

  constructor() {
    this.interrupts = {}
    this.resumeResponses = undefined
    this.activated = false
    this.pendingToolExecution = undefined
    this.context = {}
  }

  /**
   * Gets the pending tool execution state with reconstructed Message and ToolResultBlock objects.
   * Returns undefined if there is no pending execution.
   */
  getPendingExecution(): { assistantMessage: Message; completedToolResults: Map<string, ToolResultBlock> } | undefined {
    if (!this.pendingToolExecution) {
      return undefined
    }

    const assistantMessage = Message.fromMessageData(this.pendingToolExecution.assistantMessageData)

    const completedToolResults = new Map<string, ToolResultBlock>()
    for (const [toolUseId, resultData] of Object.entries(this.pendingToolExecution.completedToolResults)) {
      completedToolResults.set(toolUseId, ToolResultBlock.fromJSON(resultData))
    }

    return { assistantMessage, completedToolResults }
  }

  /**
   * Sets the pending tool execution state.
   */
  setPendingToolExecution(pending: PendingToolExecution): void {
    this.pendingToolExecution = pending
  }

  /**
   * Clears the pending tool execution state.
   */
  clearPendingToolExecution(): void {
    this.pendingToolExecution = undefined
  }

  /**
   * Returns the list of interrupts as an array.
   */
  getInterruptsList(): Interrupt[] {
    return Object.values(this.interrupts)
  }

  /**
   * Returns all interrupts that have no response (i.e., were raised but not yet answered).
   */
  getUnansweredInterrupts(): Interrupt[] {
    return Object.values(this.interrupts).filter((interrupt) => interrupt.response === undefined)
  }

  /**
   * Returns the first interrupt that has no response (i.e., was raised but not yet answered).
   */
  getUnansweredInterrupt(): Interrupt | undefined {
    for (const interrupt of Object.values(this.interrupts)) {
      if (interrupt.response === undefined) {
        return interrupt
      }
    }
    return undefined
  }

  /**
   * Activates the interrupt state.
   */
  activate(): void {
    this.activated = true
  }

  /**
   * Deactivates the interrupt state and clears all interrupts and context.
   */
  deactivate(): void {
    this.interrupts = {}
    this.resumeResponses = undefined
    this.activated = false
    this.pendingToolExecution = undefined
    this.context = {}
  }

  /**
   * Configures the interrupt state for resuming from an interrupt.
   * Populates interrupt responses from the provided content blocks.
   *
   * @param responses - Array of interrupt response content blocks
   * @throws Error if an interrupt ID is not found
   */
  resume(responses: InterruptResponseContent[]): void {
    if (!this.activated) {
      return
    }

    for (const content of responses) {
      const interruptId = content.interruptResponse.interruptId
      const response = content.interruptResponse.response

      const interrupt = this.interrupts[interruptId]
      if (!interrupt) {
        throw new Error(`interrupt_id=<${interruptId}> | no interrupt found`)
      }

      interrupt.response = response
    }

    this.resumeResponses = responses
  }

  /**
   * Gets or creates an interrupt with the given ID.
   * If the interrupt already exists, returns it (potentially with a response).
   * If a preemptive response is provided and the interrupt is new, the response
   * is stored on the interrupt so it returns immediately without halting execution.
   *
   * @param id - Unique identifier for the interrupt
   * @param name - User-defined name for the interrupt
   * @param reason - Optional reason for the interrupt
   * @param response - Optional preemptive response to skip the interrupt
   * @param source - Where the interrupt was raised from (tool or hook callback)
   * @returns The interrupt (may have a response if resuming or preemptive)
   */
  getOrCreateInterrupt(
    id: string,
    name: string,
    reason?: JSONValue,
    response?: JSONValue,
    source?: InterruptSource
  ): Interrupt {
    const existing = this.interrupts[id]
    if (existing) {
      return existing
    }

    const interrupt = new Interrupt({
      id,
      name,
      ...(reason !== undefined && { reason }),
      ...(response !== undefined && { response }),
      ...(source !== undefined && { source }),
    })
    this.interrupts[id] = interrupt
    return interrupt
  }

  /**
   * Register an existing Interrupt instance in the state, or return the already-registered one.
   */
  registerInterrupt(interrupt: Interrupt): Interrupt {
    return this.getOrCreateInterrupt(interrupt.id, interrupt.name, interrupt.reason, undefined, interrupt.source)
  }

  /**
   * Serializes the interrupt state to a JSON-compatible object.
   */
  toJSON(): InterruptStateData {
    const interrupts: Record<string, { id: string; name: string; reason?: JSONValue; response?: JSONValue }> = {}
    for (const [id, interrupt] of Object.entries(this.interrupts)) {
      interrupts[id] = interrupt.toJSON()
    }

    return {
      interrupts,
      ...(this.resumeResponses && { resumeResponses: this.resumeResponses }),
      activated: this.activated,
      ...(this.pendingToolExecution && { pendingToolExecution: this.pendingToolExecution }),
      ...(Object.keys(this.context).length > 0 && { context: this.context }),
    }
  }

  /**
   * Creates an InterruptState instance from a JSON object.
   *
   * @param data - JSON data to deserialize
   * @returns InterruptState instance
   */
  static fromJSON(data: InterruptStateData): InterruptState {
    const state = new InterruptState()
    state.activated = data.activated

    for (const [id, interruptData] of Object.entries(data.interrupts)) {
      state.interrupts[id] = Interrupt.fromJSON(interruptData)
    }

    if (data.resumeResponses) {
      state.resumeResponses = data.resumeResponses.map((r) => InterruptResponseContent.fromJSON(r))
    }

    if (data.pendingToolExecution) {
      state.pendingToolExecution = data.pendingToolExecution
    }

    if (data.context) {
      state.context = data.context
    }

    return state
  }
}

/**
 * Persists a nested agent's interrupted turn in its parent's interrupt state, scoped to one tool call.
 *
 * @internal
 */
export class NestedInterruptState {
  /** Prefix that scopes nested interrupt ids to this tool call on the parent. */
  readonly prefix: string

  /**
   * @param state - The parent's interrupt state
   * @param key - Key in `state.context` the interrupted turns are stored under
   * @param namespace - Tool-specific id namespace, e.g. `subagent`
   * @param toolUseId - Id of the tool call that runs the nested agent
   */
  constructor(
    private readonly _state: InterruptState,
    private readonly _key: string,
    namespace: string,
    private readonly _toolUseId: string
  ) {
    this.prefix = `${namespace}:${encodeURIComponent(_toolUseId)}:`
  }

  /** Whether the parent is holding an interrupt raised by this tool call. */
  isResuming(): boolean {
    return this._state.activated && Object.keys(this._state.interrupts).some((id) => id.startsWith(this.prefix))
  }

  /**
   * Store the agent's interrupted turn and register its interrupts on the parent.
   * Registered here because the concurrent executor keeps only one InterruptError per batch.
   *
   * @param agent - The interrupted nested agent
   * @param interrupts - The interrupts the agent stopped on
   * @returns The parent-scoped interrupts to throw
   */
  store(agent: Agent, interrupts: Interrupt[]): Interrupt[] {
    const turns = (this._state.context[this._key] ??= {}) as Record<string, JSONValue>
    turns[this._toolUseId] = deepCopy(agent.takeSnapshot({ preset: 'session' }))
    return interrupts.map((interrupt) =>
      this._state.registerInterrupt(
        new Interrupt({
          id: `${this.prefix}${interrupt.id}`,
          name: interrupt.name,
          ...(interrupt.reason !== undefined && { reason: interrupt.reason }),
          source: 'tool',
        })
      )
    )
  }

  /**
   * The responses supplied on this resume for this tool call, mapped back to the nested agent's ids.
   *
   * @returns The nested agent's interrupt responses
   * @throws InterruptError with this call's unanswered interrupts if none were answered
   */
  responses(): InterruptResponseContent[] {
    const responses = (this._state.resumeResponses ?? [])
      .filter((r) => r.interruptResponse.interruptId.startsWith(this.prefix))
      .map(
        (r) =>
          new InterruptResponseContent({
            interruptId: r.interruptResponse.interruptId.slice(this.prefix.length),
            response: r.interruptResponse.response,
          })
      )
    if (responses.length === 0) {
      throw new InterruptError(
        Object.values(this._state.interrupts).filter((i) => i.id.startsWith(this.prefix) && i.response === undefined)
      )
    }
    return responses
  }

  /**
   * Load the stored turn into `agent`, if there is one.
   *
   * @param agent - The nested agent to restore
   * @throws Error if the stored turn cannot be loaded; it stays stored, see {@link NestedInterruptState.unanswer}
   */
  restore(agent: Agent): void {
    const turn = this._turns()?.[this._toolUseId]
    if (turn !== undefined) {
      agent.loadSnapshot(turn as unknown as Snapshot)
    }
  }

  /**
   * Mark the stored turn's still-awaited interrupts unanswered, keeping the turn stored.
   *
   * @returns The interrupts to raise again; empty if there is no readable stored turn
   */
  unanswer(): Interrupt[] {
    const turn = this._turns()?.[this._toolUseId]
    const awaited = turn === undefined ? undefined : awaitedInterruptIds(turn)
    if (awaited === undefined) {
      logger.error(`tool_use_id=<${this._toolUseId}> | stored turn has no readable interrupt state to raise again`)
      return []
    }
    const pending = Object.values(this._state.interrupts).filter(
      (i) => i.id.startsWith(this.prefix) && awaited.includes(i.id.slice(this.prefix.length))
    )
    for (const interrupt of pending) delete interrupt.response
    return pending
  }

  /** Drop the stored turn once the nested agent is done with it. */
  clear(): void {
    delete this._turns()?.[this._toolUseId]
  }

  private _turns(): Record<string, JSONValue> | undefined {
    return this._state.context[this._key] as Record<string, JSONValue> | undefined
  }
}

/** Ids of the interrupts a stored turn is waiting on, or undefined if its interrupt state is unreadable. */
function awaitedInterruptIds(turn: JSONValue): string[] | undefined {
  try {
    const data = (turn as unknown as Snapshot).data.interrupts as unknown as InterruptStateData
    return Object.keys(InterruptState.fromJSON(data).interrupts)
  } catch {
    return undefined
  }
}

/**
 * Interface for objects that support human-in-the-loop interrupts.
 * Implemented by hook events and tool contexts that can pause agent execution.
 */
export interface Interruptible {
  interrupt<T = JSONValue>(params: InterruptParams): T
}

/**
 * Shared interrupt logic that accesses the agent's interrupt state to register or resume an interrupt.
 *
 * @param agent - The agent whose interrupt state to access
 * @param interruptId - Unique identifier for this interrupt instance
 * @param params - Interrupt parameters including name and optional reason
 * @param source - Where the interrupt was raised from (tool callback vs hook callback)
 * @returns The user's response when resuming from an interrupt
 * @throws InterruptError when no response is available (first invocation)
 *
 * @internal
 */
export function interruptFromAgent<T>(
  agent: LocalAgent,
  interruptId: string,
  params: InterruptParams,
  source: InterruptSource
): T {
  const interruptState = (agent as unknown as { _interruptState?: InterruptState })._interruptState
  if (!interruptState) {
    throw new Error('Interrupt state not available')
  }

  const interrupt = interruptState.getOrCreateInterrupt(
    interruptId,
    params.name,
    params.reason,
    params.response,
    source
  )

  if (interrupt.response !== undefined) {
    return interrupt.response as T
  }

  throw new InterruptError(interrupt)
}

/**
 * Interrupt-or-resume helper for multi-agent hooks where interrupts live on a per-node
 * `Interrupt[]` list rather than on an agent's `InterruptState`. Mirrors the
 * {@link interruptFromAgent} contract: returns the response if the interrupt already
 * has one (resume path), otherwise records a new interrupt and throws `InterruptError`.
 *
 * @internal
 */
export function interruptFromMultiAgentNode<T>(
  interrupts: Interrupt[],
  interruptId: string,
  params: InterruptParams,
  source: InterruptSource
): T {
  const existing = interrupts.find((i) => i.id === interruptId)
  if (existing?.response !== undefined) {
    return existing.response as T
  }

  const interrupt =
    existing ??
    new Interrupt({
      id: interruptId,
      name: params.name,
      ...(params.reason !== undefined && { reason: params.reason }),
      ...(params.response !== undefined && { response: params.response }),
      source,
    })

  if (!existing) {
    interrupts.push(interrupt)
    if (interrupt.response !== undefined) {
      return interrupt.response as T
    }
  }

  throw new InterruptError(interrupt)
}
