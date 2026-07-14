import type { ReactNode, CSSProperties } from 'react'
import { getAlertColor, getHealthColor } from '../../../../utils/Anomaly_Health_Monitering/presentation'

// --- KPI Card ---
export function KpiCard({
  label, value, sub, color = '#06b6d4', icon,
}: {
  label: string; value: string | number; sub?: string; color?: string; icon?: ReactNode
}) {
  return (
    <div className="glass p-5 flex flex-col gap-3 hover:border-[rgba(6,182,212,0.3)] transition-colors">
      <div className="flex items-center justify-between">
        <span style={{ color }} className="text-xs font-mono uppercase tracking-widest opacity-80">{label}</span>
        {icon && <span style={{ color }} className="opacity-60">{icon}</span>}
      </div>
      <div className="font-display font-bold text-2xl text-slate-100 tracking-tight">{value}</div>
      {sub && <div className="text-xs text-slate-500">{sub}</div>}
    </div>
  )
}

// --- Circular SVG Gauge ---
export function CircularGauge({
  value, max = 100, size = 140, color, label, sub,
}: {
  value: number; max?: number; size?: number; color: string; label?: string; sub?: string
}) {
  const R = (size - 24) / 2
  const cx = size / 2, cy = size / 2
  const arc = 270
  const startAngle = -225
  const pct = Math.min(1, Math.max(0, value / max))
  const toRad = (d: number) => (d * Math.PI) / 180
  const x1 = cx + R * Math.cos(toRad(startAngle))
  const y1 = cy + R * Math.sin(toRad(startAngle))
  const endAngle = startAngle + arc * pct
  const x2 = cx + R * Math.cos(toRad(endAngle))
  const y2 = cy + R * Math.sin(toRad(endAngle))
  const la = arc * pct > 180 ? 1 : 0

  const bgX2 = cx + R * Math.cos(toRad(startAngle + arc))
  const bgY2 = cy + R * Math.sin(toRad(startAngle + arc))

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <path
          d={`M ${x1} ${y1} A ${R} ${R} 0 1 1 ${bgX2} ${bgY2}`}
          fill="none" stroke="rgba(30,60,100,0.5)" strokeWidth="8" strokeLinecap="round"
        />
        {pct > 0 && (
          <path
            d={`M ${x1} ${y1} A ${R} ${R} 0 ${la} 1 ${x2} ${y2}`}
            fill="none" stroke={color} strokeWidth="8" strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 6px ${color}60)` }}
          />
        )}
        <text x={cx} y={cy + 4} textAnchor="middle" fill="white" fontSize="20" fontWeight="700" fontFamily="'Plus Jakarta Sans', sans-serif">
          {value >= 1 ? value.toFixed(0) : (value * 100).toFixed(0)}
        </text>
        {sub && (
          <text x={cx} y={cy + 20} textAnchor="middle" fill="#64748b" fontSize="9" fontFamily="'JetBrains Mono', monospace">
            {sub}
          </text>
        )}
      </svg>
      {label && <div className="text-xs text-slate-400 text-center">{label}</div>}
    </div>
  )
}

// --- Badge ---
export function Badge({
  label, type,
}: {
  label: string; type?: 'alert' | 'health' | 'state' | 'success' | 'neutral' | 'purple'
}) {
  let bg = 'rgba(100,116,139,0.2)', border = 'rgba(100,116,139,0.4)', text = '#94a3b8'

  if (type === 'alert') {
    const c = getAlertColor(label)
    bg = `${c}22`; border = `${c}55`; text = c
  } else if (type === 'health') {
    const c = getHealthColor(label)
    bg = `${c}22`; border = `${c}55`; text = c
  } else if (type === 'success') {
    bg = '#22c55e22'; border = '#22c55e55'; text = '#22c55e'
  } else if (type === 'purple') {
    bg = '#a855f722'; border = '#a855f755'; text = '#a855f7'
  }

  return (
    <span
      className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold font-mono"
      style={{ background: bg, border: `1px solid ${border}`, color: text }}
    >
      {label}
    </span>
  )
}

// --- Status Dot ---
export function StatusDot({ color, pulse = false }: { color: string; pulse?: boolean }) {
  return (
    <span className="relative inline-flex">
      <span
        className={pulse ? 'status-pulse' : ''}
        style={{ display: 'block', width: 8, height: 8, borderRadius: '50%', background: color, boxShadow: `0 0 6px ${color}` }}
      />
    </span>
  )
}

// --- Section Header ---
export function SectionHeader({ title, sub, action }: { title: string; sub?: string; action?: ReactNode }) {
  return (
    <div className="flex items-start justify-between mb-6">
      <div>
        <h2 className="font-display font-bold text-xl text-slate-100">{title}</h2>
        {sub && <p className="text-sm text-slate-500 mt-0.5">{sub}</p>}
      </div>
      {action}
    </div>
  )
}

// --- Page Header ---
export function PageHeader({
  title, subtitle, breadcrumb, actions,
}: { title: string; subtitle?: string; breadcrumb?: string; actions?: ReactNode }) {
  return (
    <div className="flex items-start justify-between mb-8">
      <div>
        {breadcrumb && <div className="text-xs font-mono text-slate-600 mb-1 uppercase tracking-wider">{breadcrumb}</div>}
        <h1 className="font-display font-bold text-2xl text-slate-100">{title}</h1>
        {subtitle && <p className="text-sm text-slate-500 mt-1 max-w-xl">{subtitle}</p>}
      </div>
      {actions && <div className="flex gap-3">{actions}</div>}
    </div>
  )
}

// --- Small metric row ---
export function MetricRow({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div className="flex items-center justify-between py-2.5 border-b border-[rgba(30,60,100,0.25)] last:border-0">
      <span className="text-sm text-slate-400">{label}</span>
      <span className="font-mono text-sm font-medium" style={{ color: color || '#e2e8f0' }}>{value}</span>
    </div>
  )
}

// --- Sensor contribution bar ---
export function SensorBar({
  sensor, contribution, rank, color = '#06b6d4',
}: { sensor: string; contribution: number; rank: number; color?: string }) {
  return (
    <div className="flex items-center gap-3 py-2">
      <span className="text-xs font-mono text-slate-600 w-4">#{rank}</span>
      <span className="font-mono text-sm text-slate-300 w-12">{sensor}</span>
      <div className="flex-1 h-2 bg-[rgba(30,60,100,0.4)] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${(contribution * 100).toFixed(1)}%`, background: color, boxShadow: `0 0 6px ${color}60` }}
        />
      </div>
      <span className="font-mono text-xs text-slate-400 w-10 text-right">{(contribution * 100).toFixed(1)}%</span>
    </div>
  )
}

// --- Score Bar ---
export function ScoreBar({ label, value, color, note }: { label: string; value: number; color: string; note?: string }) {
  return (
    <div className="mb-4">
      <div className="flex justify-between mb-1.5">
        <span className="text-sm text-slate-400">{label}</span>
        <span className="font-mono text-sm font-semibold" style={{ color }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-[rgba(30,60,100,0.4)] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${value * 100}%`, background: `linear-gradient(90deg, ${color}88, ${color})`, boxShadow: `0 0 8px ${color}50` }}
        />
      </div>
      {note && <div className="text-xs text-slate-600 mt-1">{note}</div>}
    </div>
  )
}

// --- Tooltip style for recharts ---
export const chartTooltipStyle: CSSProperties = {
  background: 'rgba(7,14,28,0.97)',
  border: '1px solid rgba(30,60,100,0.6)',
  borderRadius: 8,
  color: '#e2e8f0',
  fontSize: 12,
  fontFamily: "'JetBrains Mono', monospace",
}

// --- Empty State ---
export function EmptyState({ message = 'No data available' }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-slate-600">
      <div className="text-4xl mb-3 opacity-30">◎</div>
      <div className="text-sm">{message}</div>
    </div>
  )
}

export function LoadingState({ message = 'Loading backend data…' }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-slate-500">
      <div className="h-7 w-7 rounded-full border-2 border-cyan-500/20 border-t-cyan-400 animate-spin mb-3" />
      <div className="text-sm font-mono">{message}</div>
    </div>
  )
}

export function ErrorState({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-14 px-6 text-center rounded-xl border border-red-500/20 bg-red-500/5">
      <div className="text-sm font-semibold text-red-400">Backend data could not be loaded</div>
      <div className="text-xs text-slate-500 max-w-2xl">{error}</div>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="px-3 py-1.5 rounded-lg text-xs font-semibold text-red-300 border border-red-500/30 hover:bg-red-500/10"
        >
          Retry
        </button>
      )}
    </div>
  )
}

// --- Disclaimer banner ---
export function Disclaimer({ text }: { text: string }) {
  return (
    <div className="flex items-start gap-2 px-4 py-3 rounded-lg text-xs text-amber-400/70 bg-amber-500/5 border border-amber-500/15 mt-4">
      <span className="mt-0.5">⚠</span>
      <span>{text}</span>
    </div>
  )
}

// --- Filter chip ---
export function FilterChip({
  label, active, color, onClick,
}: { label: string; active: boolean; color?: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="px-3 py-1.5 rounded-full text-xs font-semibold transition-all"
      style={{
        background: active ? (color ? `${color}22` : 'rgba(6,182,212,0.15)') : 'rgba(30,60,100,0.2)',
        border: `1px solid ${active ? (color || '#06b6d4') + '55' : 'rgba(30,60,100,0.3)'}`,
        color: active ? (color || '#06b6d4') : '#64748b',
      }}
    >
      {label}
    </button>
  )
}

// --- Button ---
export function Btn({
  children, variant = 'primary', size = 'md', onClick, type = 'button', disabled,
}: {
  children: ReactNode; variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg'; onClick?: () => void; type?: 'button' | 'submit'; disabled?: boolean
}) {
  const base = 'inline-flex items-center gap-2 font-semibold rounded-lg transition-all cursor-pointer disabled:opacity-40'
  const sizes = { sm: 'px-3 py-1.5 text-xs', md: 'px-4 py-2 text-sm', lg: 'px-6 py-2.5 text-sm' }
  const variants = {
    primary: 'bg-cyan-500 text-[#040a16] hover:bg-cyan-400 shadow-[0_0_16px_rgba(6,182,212,0.3)]',
    secondary: 'bg-[rgba(30,60,100,0.4)] text-slate-300 border border-[rgba(30,60,100,0.6)] hover:bg-[rgba(30,60,100,0.6)]',
    ghost: 'text-slate-400 hover:text-slate-200 hover:bg-[rgba(30,60,100,0.25)]',
    danger: 'bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30',
  }
  return (
    <button type={type} onClick={onClick} disabled={disabled} className={`${base} ${sizes[size]} ${variants[variant]}`}>
      {children}
    </button>
  )
}
