import { describe, expect, it } from 'vitest'
import { ROOT_CONTEXT } from '@opentelemetry/api'
import { runInContext } from '../utils.js'

describe('runInContext', () => {
  it('forwards yields, return value, consumer throws, and early-return cleanup to the wrapped generator', async () => {
    const log: string[] = []
    async function* inner(): AsyncGenerator<number, string, undefined> {
      try {
        yield 1
      } catch (error) {
        log.push(`caught ${(error as Error).message}`)
      }
      try {
        yield 2
        return 'done'
      } finally {
        log.push('cleanup')
      }
    }

    const completed = runInContext(ROOT_CONTEXT, inner())
    expect(await completed.next()).toStrictEqual({ value: 1, done: false })
    expect(await completed.throw(new Error('boom'))).toStrictEqual({ value: 2, done: false })
    expect(await completed.next()).toStrictEqual({ value: 'done', done: true })

    const broken = runInContext(ROOT_CONTEXT, inner())
    await broken.next()
    await broken.next()
    await broken.return(undefined as never)

    expect(log).toStrictEqual(['caught boom', 'cleanup', 'cleanup'])
  })
})
