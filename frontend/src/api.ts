// Mirrors the response models of POST /agent in src/offers/api.py.

export interface Offer {
  title: string
  company: string
  city: string
  salary_min: number | null
  salary_max: number | null
  remote: boolean
  skills: string[]
  skills_nice_to_have: string[]
  description: string
}

export interface AgentUsage {
  llm_calls: number
  input_tokens: number
  output_tokens: number
  thinking_tokens: number
}

export interface AgentResponse {
  answer: string
  offers: Offer[]
  /** Tools the agent called in each step; the last step writes the answer. */
  steps: string[][]
  usage: AgentUsage
}

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

export async function askAgent(question: string): Promise<AgentResponse> {
  let response: Response
  try {
    response = await fetch('/agent', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    })
  } catch {
    throw new ApiError(0, "Can't reach the server. Check your connection and try again.")
  }
  if (!response.ok) {
    throw new ApiError(response.status, await errorMessage(response))
  }
  return (await response.json()) as AgentResponse
}

async function errorMessage(response: Response): Promise<string> {
  const body: unknown = await response.json().catch(() => null)
  const detail = (body as { detail?: unknown } | null)?.detail
  if (response.status === 422) {
    return 'Write a question of at least 3 characters.'
  }
  if (typeof detail === 'string') {
    return `${detail}. Try again in a moment.`
  }
  return `The server answered with an error (${response.status}). Try again in a moment.`
}
