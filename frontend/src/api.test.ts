import { afterEach, expect, it, vi } from 'vitest'

import { ApiError, askAgent } from './api.ts'

afterEach(() => {
  vi.restoreAllMocks()
})

it('explains a rejected question instead of showing validation details', async () => {
  const body = { detail: [{ type: 'string_too_short', loc: ['body', 'question'] }] }
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(
    new Response(JSON.stringify(body), { status: 422 }),
  )

  const error = await askAgent('?').catch((caught: unknown) => caught)

  expect(error).toBeInstanceOf(ApiError)
  expect(error).toMatchObject({
    status: 422,
    message: 'Write a question of at least 3 characters.',
  })
})

it('falls back to the status code when the error has no detail', async () => {
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(
    new Response('Internal Server Error', { status: 500 }),
  )

  await expect(askAgent('Remote offers?')).rejects.toThrow(
    'The server answered with an error (500). Try again in a moment.',
  )
})
