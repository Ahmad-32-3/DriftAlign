import { COMPARISON } from '../../data'

const TONE: Record<(typeof COMPARISON)[number]['tone'], string> = {
  good: 'var(--good)',
  bad: 'var(--bad)',
  muted: 'var(--fg-low)',
}

export function ResultBars() {
  const max = 100
  const w = 640
  const rowH = 36
  const left = 148
  const h = COMPARISON.length * rowH + 16
  return (
    <svg className="chart-svg" viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Target success percent with OT, without OT, majority, and chance">
      {COMPARISON.map((row, i) => {
        const y = 8 + i * rowH
        const bar = ((w - left - 72) * row.pct) / max
        return (
          <g key={row.label}>
            <text x={0} y={y + 16} fill="var(--fg)" fontSize="13">
              {row.label}
            </text>
            <rect
              className="bar-grow"
              x={left}
              y={y + 4}
              width={bar}
              height={18}
              fill={TONE[row.tone]}
              style={{ animationDelay: `${i * 0.08}s` }}
            />
            <text x={left + bar + 8} y={y + 17} fill="var(--fg-hi)" fontSize="13">
              {row.pct.toFixed(1)}%
            </text>
          </g>
        )
      })}
    </svg>
  )
}
