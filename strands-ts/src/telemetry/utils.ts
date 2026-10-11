/**
 * Shared telemetry utilities.
 */

import { context, type Context } from '@opentelemetry/api'

const DEFAULT_SERVICE_NAME = 'strands-agents'

/**
 * Get the service name, respecting the OTEL_SERVICE_NAME environment variable.
 *
 * @returns The service name from OTEL_SERVICE_NAME or the default 'strands-agents'
 */
export function getServiceName(): string {
  return globalThis.process?.env?.OTEL_SERVICE_NAME || DEFAULT_SERVICE_NAME
}

/**
 * Drives an async generator so every step (next/throw/return) runs inside the given OTel context.
 *
 * Async generators resume in the caller's context, so wrapping only creation is not enough.
 *
 * @param ctx - Context to make active while the generator runs
 * @param generator - Generator to drive
 * @returns A generator that forwards the events and return value of `generator`
 */
export async function* runInContext<T, R>(
  ctx: Context,
  generator: AsyncGenerator<T, R, undefined>
): AsyncGenerator<T, R, undefined> {
  let step = await context.with(ctx, () => generator.next())
  try {
    while (!step.done) {
      try {
        yield step.value
      } catch (error) {
        step = await context.with(ctx, () => generator.throw(error))
        continue
      }
      step = await context.with(ctx, () => generator.next())
    }
    return step.value
  } finally {
    // Consumer broke out early: let the inner generator run its cleanup in the same context.
    if (!step.done) await context.with(ctx, () => generator.return(undefined as never))
  }
}
