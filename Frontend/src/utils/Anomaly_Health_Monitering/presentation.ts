const ALERT_COLORS: Record<string, string> = {
  normal: '#22c55e',
  watch: '#f59e0b',
  warning: '#f97316',
  critical: '#ef4444',
}

const HEALTH_COLORS: Record<string, string> = {
  healthy: '#22c55e',
  degrading: '#f59e0b',
  warning: '#f97316',
  critical: '#ef4444',
}

export function getAlertColor(level?: string | null): string {
  return ALERT_COLORS[String(level ?? '').toLowerCase()] ?? '#64748b'
}

export function getHealthColor(state?: string | null): string {
  return HEALTH_COLORS[String(state ?? '').toLowerCase()] ?? '#64748b'
}

export function getHealthIndexColor(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return '#64748b'
  if (value >= 85) return '#22c55e'
  if (value >= 65) return '#f59e0b'
  if (value >= 40) return '#f97316'
  return '#ef4444'
}

export function formatCount(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return 'N/A'
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value)
}

export function formatPercent(value?: number | null, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return 'N/A'
  return `${(value * 100).toFixed(digits)}%`
}

export function formatMetric(value?: number | null, digits = 3): string {
  if (value == null || !Number.isFinite(value)) return 'N/A'
  return value.toFixed(digits)
}

export function humanize(value?: string | null): string {
  if (!value) return 'Not available'
  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase())
}

export function toFiniteNumber(value: unknown): number | null {
  const numberValue = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(numberValue) ? numberValue : null
}
