import { type FormEvent, useEffect, useState } from 'react'

import { AgentPath } from './AgentPath.tsx'
import { type AgentResponse, ApiError, askAgent } from './api.ts'
import { OfferList } from './OfferList.tsx'

const EXAMPLES = [
  'Is the average salary higher in Krakow or in Warsaw?',
  'Where in Poznan will a junior get support from a mentor?',
  'Which remote offers include a training budget?',
  'What is the average salary of the jobs whose ads explicitly offer flexible working hours?',
]

const MIN_QUESTION_LENGTH = 3

type State =
  | { status: 'idle' }
  | { status: 'asking'; startedAt: number }
  | { status: 'answered'; result: AgentResponse }
  | { status: 'failed'; message: string }

export default function App() {
  const [question, setQuestion] = useState('')
  const [state, setState] = useState<State>({ status: 'idle' })
  const asking = state.status === 'asking'

  async function ask(text: string) {
    const trimmed = text.trim()
    if (trimmed.length < MIN_QUESTION_LENGTH || asking) {
      return
    }
    setQuestion(trimmed)
    setState({ status: 'asking', startedAt: performance.now() })
    try {
      setState({ status: 'answered', result: await askAgent(trimmed) })
    } catch (error) {
      const message = error instanceof ApiError ? error.message : 'Something went wrong. Try again.'
      setState({ status: 'failed', message })
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    void ask(question)
  }

  return (
    <div className="page">
      <header className="intro">
        <h1>Ask about 18 tech job offers</h1>
        <p>
          An AI agent answers from synthetic job offers in five Polish cities. It decides which
          tools to use: it filters the offers, searches the text of the ads and computes salary
          statistics. Below the answer you can see each step it took.
        </p>
      </header>

      <form className="ask" onSubmit={handleSubmit}>
        <label htmlFor="question">Your question</label>
        <div className="ask-row">
          <input
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Which offers in Warsaw allow remote work?"
            autoComplete="off"
          />
          <button type="submit" disabled={asking || question.trim().length < MIN_QUESTION_LENGTH}>
            Ask
          </button>
        </div>
      </form>

      <div className="examples">
        <p>Or try one of these:</p>
        <ul>
          {EXAMPLES.map((example) => (
            <li key={example}>
              <button type="button" disabled={asking} onClick={() => void ask(example)}>
                {example}
              </button>
            </li>
          ))}
        </ul>
      </div>

      <main className="result" aria-live="polite">
        {state.status === 'asking' && (
          <p className="status">
            The agent is working on it (<Elapsed since={state.startedAt} />)
          </p>
        )}
        {state.status === 'failed' && (
          <p className="error" role="alert">
            {state.message}
          </p>
        )}
        {state.status === 'answered' && (
          <>
            <section aria-labelledby="answer-heading">
              <h2 id="answer-heading">Answer</h2>
              <p className="answer">{state.result.answer}</p>
            </section>
            <OfferList offers={state.result.offers} />
            <AgentPath steps={state.result.steps} usage={state.result.usage} />
          </>
        )}
      </main>

      <footer className="footer">
        <a href="https://github.com/BaseAdam/job-offers-ai">Source code and evals on GitHub</a>
      </footer>
    </div>
  )
}

function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(() => performance.now())

  useEffect(() => {
    const timer = setInterval(() => setNow(performance.now()), 100)
    return () => clearInterval(timer)
  }, [])

  return <span aria-hidden="true">{((now - since) / 1000).toFixed(1)} s</span>
}
