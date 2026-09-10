import { ABLATION } from '../../data'

export function AblationBars() {
  const rows = ABLATION.map((r) => ({
    label: `ε ${r.eps} · ${r.cost} · n ${r.n}`,
    ot: r.ot_pct,
    noOt: r.no_ot_pct,
  }))
  const w = 640
  const rowH = 44
  const left = 210
  const barW = w - left - 80
  const h = rows.length * rowH + 28
  return (
    <svg className="chart-svg" viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Ablation of epsilon, cost, and sample cap, OT versus no OT">
      {rows.map((row, i) => {
        const y = 20 + i * rowH
        return (
          <g key={row.label}>
            <text x={0} y={y + 14} fill="var(--fg)" fontSize="11">
              {row.label}
            </text>
            <rect className="bar-grow" x={left} y={y} width={(barW * row.ot) / 100} height={10} fill="var(--good)" />
            <rect className="bar-grow" x={left} y={y + 14} width={(barW * row.noOt) / 100} height={10} fill="var(--bad)" />
            <text x={left + (barW * row.ot) / 100 + 6} y={y + 10} fill="var(--fg-hi)" fontSize="11">
              {row.ot.toFixed(0)}
            </text>
          </g>
        )
      })}
    </svg>
  )
}
