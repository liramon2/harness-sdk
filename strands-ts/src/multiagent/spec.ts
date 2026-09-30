/**
 * Multi-agent specification and resolution.
 *
 * Provides the authority-mode system for model-driven multi-agent patterns, where the
 * model configures child agents at runtime. The developer sets axis policies that shape
 * the parameters the model sees and govern what values it can supply:
 *
 * - {@link Fixed}   — developer-pinned value; hidden from the model.
 * - {@link Inherit} — value taken from the parent agent; hidden from the model.
 * - {@link Open}    — model supplies a free-form value.
 * - {@link Choice}  — model picks from a developer-supplied set.
 *
 * {@link resolveSpec} merges model-supplied arguments, preset defaults, and axis policies
 * into a fully resolved {@link AgentSpec} used to build child agents.  {@link Inherit} axes
 * resolve to `undefined` (meaning "inherit all"); the class is a self-documenting marker.
 */

import type { Model } from '../models/model.js'
import type { ModelRouter } from '../models/routing/router.js'
import { Agent } from '../agent/agent.js'
import type { McpClient } from '../mcp/index.js'
import type { Tool } from '../tools/tool.js'
import { McpTool } from '../tools/mcp-tool.js'

/** Valid values for the `context` axis. */
export type ContextMode = 'none' | 'all' | 'no_tools'

/** Turns a resolved spec into a child agent, built the way the parent was. */
export type AgentBuilder = (spec: AgentSpec) => Agent

/**
 * Sentinel so `undefined` can be a legitimate {@link Option.value}.
 *
 * @internal
 */
export const UNSET: unique symbol = Symbol('UNSET')

/** The developer pins the value; the axis contributes no model-facing parameter. */
export class Fixed {
  readonly value: unknown

  /**
   * @param value - The pinned value.
   */
  constructor(value: unknown = undefined) {
    this.value = value
    Object.freeze(this)
  }
}

/** The child takes the parent's value; the axis contributes no model-facing parameter. */
export class Inherit {
  constructor() {
    Object.freeze(this)
  }
}

/** The model writes the value freely; the axis contributes a string parameter. */
export class Open {
  constructor() {
    Object.freeze(this)
  }
}

/**
 * One selectable option inside a {@link Choice}.
 *
 * `name` is what the model picks (the enum entry and the key we map back on); `value` is what
 * that choice resolves to, defaulting to `name` so a string-valued axis needs only a name.
 * `description` guides the model.
 */
export class Option {
  readonly name: string
  readonly value: unknown
  readonly description: string

  /**
   * @param name - Model-facing label for this option.
   * @param value - Resolved value; defaults to {@link UNSET}, filled from `name` by {@link Choice.normalized}.
   * @param description - Guidance shown to the model.
   */
  constructor(name: string, value: unknown = UNSET, description: string = '') {
    this.name = name
    this.value = value
    this.description = description
    Object.freeze(this)
  }
}

/**
 * The model picks from a developer-supplied set by name.
 *
 * The axis contributes an enum parameter, or an `array` of enum when `multiple` is set
 * (letting the model pick several).
 *
 * Each entry in `options` is a bare name or an {@link Option}. The picked name maps back to
 * the option's `value`; descriptions render into the parameter's `description` (JSON Schema
 * has no per-enum-value doc), the same way {@link Preset} descriptions surface under
 * `agent_type`. For `tools` the chosen set is re-validated at call time, so a delegate can
 * never exceed the parent's tools.
 */
export class Choice {
  readonly options: readonly (string | Option)[]
  readonly multiple: boolean

  /**
   * @param options - Bare names or {@link Option} instances the model can pick from.
   * @param multiple - When `true`, the axis produces an array enum allowing multiple selections.
   */
  constructor(options: readonly (string | Option)[], multiple: boolean = false) {
    this.options = options
    this.multiple = multiple
    Object.freeze(this)
  }

  /**
   * Returns the options as {@link Option} instances with `value` resolved, wrapping bare names.
   *
   * @returns Normalized options with {@link UNSET} values filled from their name.
   */
  normalized(): Option[] {
    const result: Option[] = []
    for (const o of this.options) {
      if (o instanceof Option) {
        result.push(o.value !== UNSET ? o : new Option(o.name, o.name, o.description))
      } else {
        const name = String(o)
        result.push(new Option(name, name))
      }
    }
    return result
  }

  /**
   * Renders this axis as a JSON Schema property for a tool's `inputSchema`.
   *
   * Produces a `string` enum (or `array` of enum when `multiple`), with per-option
   * descriptions folded into the property's `description`.
   *
   * @param description - Base description prepended before any per-option detail.
   * @returns A JSON Schema property object.
   */
  toSchemaProperty(description: string = ''): Record<string, unknown> {
    const options = this.normalized()
    const values = options.map((o) => o.name)
    const lines = options.filter((o) => o.description).map((o) => `- ${o.name}: ${o.description}`)
    let desc = description
    if (lines.length > 0) {
      const block = 'Options:\n' + lines.join('\n')
      desc = description ? `${description}\n${block}` : block
    }
    const prop: Record<string, unknown> = this.multiple
      ? { type: 'array', items: { type: 'string', enum: values } }
      : { type: 'string', enum: values }
    if (desc) {
      prop['description'] = desc
    }
    return prop
  }

  /**
   * Returns the resolved value behind `name`, or `name` itself if unknown (passthrough).
   *
   * @param name - The option name selected by the model.
   * @returns The mapped value, or `name` when no matching option exists.
   */
  valueFor(name: string): unknown {
    for (const o of this.normalized()) {
      if (o.name === name) {
        return o.value
      }
    }
    return name
  }
}

/** Options for constructing a {@link Preset}. */
export interface PresetOptions {
  instructions?: string
  tools?: readonly string[]
  model?: Model | ModelRouter | string
  context?: ContextMode
  lastMessages?: number
  description?: string
}

/**
 * A named role: a partially applied child configuration selected via `agent_type`.
 */
export class Preset {
  readonly instructions: string | undefined
  readonly tools: readonly string[] | undefined
  readonly model: Model | ModelRouter | string | undefined
  readonly context: ContextMode
  readonly lastMessages: number | undefined
  readonly description: string

  /**
   * @param options - Partial preset configuration.
   */
  constructor(options: PresetOptions = {}) {
    this.instructions = options.instructions
    this.tools = options.tools
    this.model = options.model
    this.context = options.context ?? 'none'
    this.lastMessages = options.lastMessages
    this.description = options.description ?? ''
    Object.freeze(this)
  }
}

/**
 * Resolved child configuration handed to the builder: config merged with the model's arguments.
 */
export class AgentSpec {
  name: string | undefined
  task: string
  agentType: string | undefined
  instructions: string | undefined
  tools: string[] | undefined
  mcpServers: string[] | undefined
  model: Model | ModelRouter | string | undefined
  context: ContextMode
  lastMessages: number | undefined

  /**
   * @param data - Partial spec fields; unset fields receive safe defaults.
   */
  constructor(data: Partial<AgentSpec> = {}) {
    this.name = data.name
    this.task = data.task ?? ''
    this.agentType = data.agentType
    this.instructions = data.instructions
    this.tools = data.tools
    this.mcpServers = data.mcpServers
    this.model = data.model
    this.context = data.context ?? 'none'
    this.lastMessages = data.lastMessages
  }
}

/**
 * Resolves a scalar axis: model-supplied value → preset → axis default.
 *
 * @param modelValue - The value the model supplied, or {@link UNSET}.
 * @param axis - The axis policy governing this parameter.
 * @param presetValue - The preset's value for this axis, if any.
 * @returns The resolved value.
 */
function resolveScalar(
  modelValue: unknown,
  axis: Open | Choice | Fixed | Inherit,
  presetValue: unknown = undefined
): unknown {
  if (modelValue !== UNSET) {
    if (axis instanceof Open) return modelValue
    if (axis instanceof Choice && axis.normalized().some((o) => o.name === modelValue)) {
      return axis.valueFor(modelValue as string)
    }
  }
  if (presetValue !== undefined) return presetValue
  if (axis instanceof Fixed) return axis.value
  return undefined
}

/**
 * Resolves a list axis: model-supplied value → preset → axis default → `undefined`.
 *
 * When the axis is a {@link Choice}, ignores values that are not valid options.
 *
 * @param modelValue - The value the model supplied, or {@link UNSET}.
 * @param axis - The axis policy governing this parameter.
 * @param presetValues - The preset's values for this axis, if any.
 * @returns The resolved list, or `undefined` to inherit from the parent.
 */
function resolveList(
  modelValue: unknown,
  axis: Choice | Fixed | Inherit,
  presetValues: readonly string[] | undefined = undefined
): string[] | undefined {
  const allowed = axis instanceof Choice ? axis.normalized().map((o) => o.name) : undefined

  // Model-supplied value
  if (modelValue !== UNSET && allowed !== undefined) {
    const choice = axis as Choice
    let requested: string[]
    if (typeof modelValue === 'string') {
      requested = [modelValue]
    } else if (Array.isArray(modelValue)) {
      requested = modelValue as string[]
    } else {
      requested = []
    }
    return requested.filter((t) => allowed.includes(t)).map((t) => choice.valueFor(t) as string)
  }

  // Preset value
  if (presetValues !== undefined) {
    if (allowed !== undefined) {
      const choice = axis as Choice
      const allowedValues = new Set(allowed.map((n) => choice.valueFor(n)))
      return presetValues.filter((t) => allowedValues.has(t))
    }
    return [...presetValues]
  }

  // All choices
  if (axis instanceof Choice && axis.multiple && allowed !== undefined) {
    return allowed.map((n) => (axis as Choice).valueFor(n) as string)
  }
  if (axis instanceof Fixed) {
    return axis.value != null ? [...(axis.value as string[])] : undefined
  }
  return undefined
}

/** Axes passed to {@link resolveSpec}. */
export interface ResolveSpecAxes {
  presets: Record<string, Preset>
  defaultPreset: string | undefined
  instructions: Open | Choice | Fixed
  /** Defaults to {@link Inherit} when omitted. */
  tools?: Choice | Fixed | Inherit
  /** Defaults to {@link Inherit} when omitted. */
  mcpServers?: Choice | Fixed | Inherit
  /** Defaults to {@link Inherit} when omitted. */
  model?: Inherit | Choice | Fixed
  /** Defaults to `Fixed('none')` when omitted. */
  context?: Fixed | Choice
}

/**
 * Combines the model's arguments, the selected preset, and the fixed axes into a spec.
 *
 * Precedence per axis: a model-supplied argument wins, then the preset's value, then the axis
 * default ({@link Fixed}/{@link Inherit}). An omitted `agent_type` falls back to the default
 * preset, so a bare `subagent(task=...)` behaves like the default role.
 *
 * @param modelInput - Key/value pairs supplied by the model's tool call.
 * @param axes - Axis policies, presets, and the default preset name.
 * @returns A fully resolved {@link AgentSpec}.
 * @throws Error if `agent_type` is provided but not found in `axes.presets`.
 */
export function resolveSpec(modelInput: Record<string, unknown>, axes: ResolveSpecAxes): AgentSpec {
  const { presets, defaultPreset, instructions } = axes
  const tools = axes.tools ?? new Inherit()
  const mcpServers = axes.mcpServers ?? new Inherit()
  const model = axes.model ?? new Inherit()
  const context = axes.context ?? new Fixed('none')

  const task = String(modelInput['task'] ?? '')
  const name = modelInput['name'] as string | undefined

  // agent_type is a closed enum: a provided value must be an exact preset name.
  const modelAgentType = modelInput['agent_type']
  const isKnownPreset = typeof modelAgentType === 'string' && Object.hasOwn(presets, modelAgentType)
  if (modelAgentType != null && !isKnownPreset) {
    throw new Error(
      `Unknown agent_type ${JSON.stringify(modelAgentType)}; valid values: ${Object.keys(presets).sort().join(', ')}.`
    )
  }

  // Ad-hoc instructions (only possible when the axis exposes one) override the default preset.
  const overridingInstructions =
    (instructions instanceof Open || instructions instanceof Choice) && 'instructions' in modelInput

  let agentType: string | undefined
  if (isKnownPreset) {
    agentType = modelAgentType as string
  } else if (!overridingInstructions) {
    agentType = defaultPreset
  }

  const preset = agentType ? presets[agentType] : undefined

  const spec = new AgentSpec({
    name: name !== undefined ? String(name) : undefined,
    task,
    agentType,
  })

  spec.instructions = resolveScalar(modelInput['instructions'] ?? UNSET, instructions, preset?.instructions) as
    string | undefined

  spec.tools = resolveList(modelInput['tools'] ?? UNSET, tools, preset?.tools)

  spec.mcpServers = resolveList(modelInput['mcp_servers'] ?? UNSET, mcpServers)

  spec.model = resolveScalar(modelInput['model'] ?? UNSET, model, preset?.model) as
    Model | ModelRouter | string | undefined

  const contextVal = resolveScalar(modelInput['context'] ?? UNSET, context, preset?.context)
  spec.context = (contextVal as ContextMode) ?? 'none'

  // last_messages — always available when context is not "none", but optional.
  // Preset provides the default; a valid model-supplied value overrides it.
  if (preset?.lastMessages !== undefined) {
    spec.lastMessages = preset.lastMessages
  }
  if ('last_messages' in modelInput) {
    const parsed = Number(modelInput['last_messages'])
    if (Number.isFinite(parsed)) {
      spec.lastMessages = parsed
    }
  }

  return spec
}

/**
 * Creates a builder that produces child agents inheriting the parent's model, tools, and
 * MCP servers.
 *
 * Used by the vended multi-agent tools when no custom builder is supplied.
 * Resolves `spec.tools` against the parent's tool registry and `spec.mcpServers`
 * against the parent's MCP clients (by `clientName`).
 *
 * @param parent - The parent agent whose resources child agents inherit.
 * @returns An {@link AgentBuilder} that creates configured child agents from a spec.
 */
export function defaultBuilder(parent: Agent): AgentBuilder {
  return (spec: AgentSpec): Agent => {
    const parentTools: Map<string, Tool> = new Map()
    const mcpClients: Map<string, McpClient> = new Map()
    for (const t of parent.toolRegistry.list()) {
      if (t instanceof McpTool) {
        // MCP tools flow through mcpServers to avoid duplicates with their client.
        const client = (t as unknown as { mcpClient: McpClient }).mcpClient
        if (client.clientName) {
          mcpClients.set(client.clientName, client)
        }
      } else {
        // Plain tools are selected by name via spec.tools.
        parentTools.set(t.name, t)
      }
    }

    const childTools: (Tool | McpClient)[] = []

    // tools=undefined means inherit all; a list means only those.
    const selectedTools =
      spec.tools === undefined ? parentTools : new Map([...parentTools].filter(([n]) => spec.tools!.includes(n)))
    for (const tool of selectedTools.values()) {
      childTools.push(tool)
    }

    // mcpServers=undefined means inherit all; a list means only those.
    const selected =
      spec.mcpServers === undefined
        ? mcpClients
        : new Map([...mcpClients].filter(([n]) => spec.mcpServers!.includes(n)))
    for (const client of selected.values()) {
      childTools.push(client)
    }

    return new Agent({
      systemPrompt: spec.instructions ?? '',
      tools: childTools,
      model: (spec.model as Model | ModelRouter | string) ?? parent.model,
      ...(spec.name !== undefined && { name: spec.name }),
    })
  }
}
