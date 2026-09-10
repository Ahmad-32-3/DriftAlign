import { CLOUD } from '../../data'

const COLORS = ['var(--accent)', 'var(--good)', 'var(--amber)']

function bounds(pts: { x: number; y: number }[]) {
  const xs = pts.map((p) => p.x)
  const ys = pts.map((p) => p.y)
  const pad = 0.8
  return {
    minX: Math.min(...xs) - pad,
    maxX: Math.max(...xs) + pad,
    minY: Math.min(...ys) - pad,
    maxY: Math.max(...ys) + pad,
  }
}

export function DomainCloud() {
  const pts = [...CLOUD.source, ...CLOUD.target]
  if (pts.length === 0) return null
  const b = bounds(pts)
  const w = 640
  const h = 280
  const sx = (x: number) => ((x - b.minX) / (b.maxX - b.minX)) * (w - 24) + 12
  const sy = (y: number) => h - 12 - ((y - b.minY) / (b.maxY - b.minY)) * (h - 24)
  return (
    <svg className="chart-svg" viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Source and target point clouds in feature space">
      {CLOUD.source.map((p, i) => (
        <circle key={`s${i}`} cx={sx(p.x)} cy={sy(p.y)} r={4} fill={COLORS[p.c] ?? COLORS[0]} opacity={0.55} />
      ))}
      {CLOUD.target.map((p, i) => (
        <rect
          key={`t${i}`}
          x={sx(p.x) - 3.5}
          y={sy(p.y) - 3.5}
          width={7}
          height={7}
          fill={COLORS[p.c] ?? COLORS[0]}
        />
      ))}
    </svg>
  )
}
