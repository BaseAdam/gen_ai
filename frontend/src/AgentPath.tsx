import type { CSSProperties } from 'react'

import type { AgentUsage } from './api.ts'
import { formatNumber } from './format.ts'

const TOOL_DESCRIPTIONS: Record<string, string> = {
  list_offers: 'Filtered the offers',
  search_job_ads: 'Searched the job ads',
  salary_stats: 'Computed salary statistics',
}

export function AgentPath({ steps, usage }: { steps: string[][]; usage: AgentUsage }) {
  const tokens = usage.input_tokens + usage.output_tokens + usage.thinking_tokens
  return (
    <section aria-labelledby="path-heading">
      <h2 id="path-heading">How the agent got there</h2>
      <ol className="route">
        {steps.map((tools, index) => (
          <li key={index} className="stop" style={{ '--stop': index } as CSSProperties}>
            {tools.length === 0 ? (
              <span className="stop-text">Wrote the answer</span>
            ) : (
              tools.map((tool, toolIndex) => (
                <span key={toolIndex} className="stop-text">
                  {TOOL_DESCRIPTIONS[tool] ?? 'Used a tool'} <code>{tool}</code>
                </span>
              ))
            )}
          </li>
        ))}
      </ol>
      <p className="usage">
        {usage.llm_calls} model {usage.llm_calls === 1 ? 'call' : 'calls'} and{' '}
        {formatNumber(tokens)} tokens, {formatNumber(usage.thinking_tokens)} of them for thinking.
      </p>
    </section>
  )
}
