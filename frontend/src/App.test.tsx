import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, expect, it, vi } from 'vitest'

import type { AgentResponse } from './api.ts'
import App from './App.tsx'

const RESULT: AgentResponse = {
  answer: 'Krakow pays more on average.',
  offers: [
    {
      title: 'Backend Engineer',
      company: 'Neptune Systems',
      city: 'Krakow',
      salary_min: 12000,
      salary_max: 17000,
      remote: true,
      skills: ['python', 'fastapi'],
      skills_nice_to_have: [],
      description: '',
    },
  ],
  steps: [['list_offers', 'salary_stats'], []],
  usage: { llm_calls: 2, input_tokens: 3000, output_tokens: 120, thinking_tokens: 80 },
}

function respondWith(status: number, body: unknown) {
  return vi
    .spyOn(globalThis, 'fetch')
    .mockResolvedValue(new Response(JSON.stringify(body), { status }))
}

afterEach(() => {
  vi.restoreAllMocks()
})

it('asks the agent and shows the answer, the offers and the path', async () => {
  const fetch = respondWith(200, RESULT)
  const user = userEvent.setup()
  render(<App />)

  await user.type(screen.getByLabelText('Your question'), 'Is Krakow better paid than Warsaw?')
  await user.click(screen.getByRole('button', { name: 'Ask' }))

  expect(await screen.findByText('Krakow pays more on average.')).toBeInTheDocument()
  expect(fetch).toHaveBeenCalledWith('/agent', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: 'Is Krakow better paid than Warsaw?' }),
  })
  const offers = screen.getByRole('region', { name: 'Matching offer' })
  expect(within(offers).getByRole('heading', { name: 'Backend Engineer' })).toBeInTheDocument()
  expect(within(offers).getByText('12,000–17,000 PLN')).toBeInTheDocument()
  expect(within(offers).getByText('Remote')).toBeInTheDocument()
  const path = screen.getByRole('region', { name: 'How the agent got there' })
  const stops = within(path).getAllByRole('listitem')
  expect(stops).toHaveLength(2)
  expect(stops[0]).toHaveTextContent('Filtered the offers list_offers')
  expect(stops[0]).toHaveTextContent('Computed salary statistics salary_stats')
  expect(stops[1]).toHaveTextContent('Wrote the answer')
  expect(
    within(path).getByText('2 model calls and 3,200 tokens, 80 of them for thinking.'),
  ).toBeInTheDocument()
})

it('asks an example question with one click', async () => {
  const fetch = respondWith(200, RESULT)
  const user = userEvent.setup()
  render(<App />)

  const example = 'Which remote offers include a training budget?'
  await user.click(screen.getByRole('button', { name: example }))

  expect(await screen.findByText(RESULT.answer)).toBeInTheDocument()
  expect(fetch.mock.calls[0][1]?.body).toBe(JSON.stringify({ question: example }))
  expect(screen.getByLabelText('Your question')).toHaveValue(example)
})

it('shows progress and blocks new questions while the agent works', async () => {
  let respond: (response: Response) => void = () => {}
  vi.spyOn(globalThis, 'fetch').mockReturnValue(
    new Promise((resolve) => {
      respond = resolve
    }),
  )
  const user = userEvent.setup()
  render(<App />)

  const ask = screen.getByRole('button', { name: 'Ask' })
  expect(ask).toBeDisabled()
  await user.type(screen.getByLabelText('Your question'), 'Remote offers?')
  await user.click(ask)

  expect(screen.getByText(/The agent is working on it/)).toBeInTheDocument()
  expect(ask).toBeDisabled()
  expect(screen.getByRole('button', { name: /training budget/ })).toBeDisabled()

  respond(new Response(JSON.stringify(RESULT)))
  expect(await screen.findByText(RESULT.answer)).toBeInTheDocument()
  expect(ask).toBeEnabled()
})

it('shows what went wrong when the API fails', async () => {
  respondWith(503, { detail: 'Database is unavailable' })
  const user = userEvent.setup()
  render(<App />)

  await user.click(screen.getByRole('button', { name: /training budget/ }))

  expect(await screen.findByRole('alert')).toHaveTextContent(
    'Database is unavailable. Try again in a moment.',
  )
})

it('says when the server cannot be reached', async () => {
  vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('Failed to fetch'))
  const user = userEvent.setup()
  render(<App />)

  await user.click(screen.getByRole('button', { name: /training budget/ }))

  expect(await screen.findByRole('alert')).toHaveTextContent("Can't reach the server.")
})
