import { useMemo } from 'react'
import {
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  Activity,
  AlertTriangle,
  RefreshCw,
  ShieldCheck,
  TrendingDown,
  Waves,
} from 'lucide-react'
import { useDashboardAnalytics } from '../../../hooks/Anomaly_Health_Monitering'
import {
  EmptyState,
  KpiCard,
  PageHeader,
  SectionHeader,
  chartTooltipStyle,
} from '../../components/ui/Anomaly_Health_Monitering/ui'

const HEALTH_COLORS: Record<string, string> = {
  Healthy: '#22c55e',
  Degrading: '#3b82f6',
  Warning: '#f59e0b',
  Critical: '#ef4444',
}

type JsonRecord = Record<string, unknown>

function asRecord(value: unknown): JsonRecord | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as JsonRecord
    : null
}

function numberAt(record: JsonRecord | null, key: string): number | undefined {
  const value = record?.[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function formatCount(value: number | undefined): string {
  if (value === undefined) return '—'
  return new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value)
}

function formatNumber(value: number | undefined, digits = 2): string {
  return value === undefined ? '—' : value.toFixed(digits)
}

function formatPercent(value: number | undefined, digits = 1): string {
  return value === undefined ? '—' : `${(value * 100).toFixed(digits)}%`
}

function healthColor(value: number | undefined): string {
  if (value === undefined) return '#64748b'
  if (value >= 85) return '#22c55e'
  if (value >= 65) return '#3b82f6'
  if (value >= 40) return '#f59e0b'
  return '#ef4444'
}

export default function HealthMonitoring() {
  const analyticsResource = useDashboardAnalytics()
  const analytics = asRecord(analyticsResource.data)
  const summaries = asRecord(analytics?.summaries)
  const dashboardSummary = asRecord(summaries?.dashboard_data_summary)
  const healthEvaluation = asRecord(summaries?.evaluate_health_summary)
  const stateCounts = asRecord(dashboardSummary?.health_state_counts)
  const dashboardAverages = asRecord(dashboardSummary?.averages)
  const splitSummary = asRecord(healthEvaluation?.split_summary)

  const stateDistribution = useMemo(() => [
    { state: 'Healthy', count: numberAt(stateCounts, 'Healthy'), color: HEALTH_COLORS.Healthy },
    { state: 'Degrading', count: numberAt(stateCounts, 'Degrading'), color: HEALTH_COLORS.Degrading },
    { state: 'Warning', count: numberAt(stateCounts, 'Warning'), color: HEALTH_COLORS.Warning },
    { state: 'Critical', count: numberAt(stateCounts, 'Critical'), color: HEALTH_COLORS.Critical },
  ].filter((item): item is { state: string; count: number; color: string } => item.count !== undefined), [stateCounts])

  const totalStateRecords = useMemo(
    () => stateDistribution.reduce((total, item) => total + item.count, 0),
    [stateDistribution],
  )

  const splitRows = useMemo(() => Object.entries(splitSummary ?? {}).flatMap(([split, value]) => {
    const details = asRecord(value)
    if (!details) return []
    return [{
      split,
      row_count: numberAt(details, 'row_count'),
      average_health_index: numberAt(details, 'average_health_index'),
      healthy_ratio: numberAt(details, 'healthy_ratio'),
      degrading_ratio: numberAt(details, 'degrading_ratio'),
      warning_ratio: numberAt(details, 'warning_ratio'),
      critical_ratio: numberAt(details, 'critical_ratio'),
      health_trend_smoothness: numberAt(details, 'health_trend_smoothness'),
      deterioration_consistency: numberAt(details, 'health_deterioration_consistency'),
      deteriorating_ratio: numberAt(details, 'deteriorating_trend_ratio'),
    }]
  }), [splitSummary])

  const averageBySplit = splitRows
    .filter((row): row is typeof row & { average_health_index: number } => row.average_health_index !== undefined)
    .map((row) => ({ split: row.split, average_health_index: row.average_health_index, color: healthColor(row.average_health_index) }))

  const deteriorationBySplit = splitRows
    .filter((row): row is typeof row & { deteriorating_ratio: number } => row.deteriorating_ratio !== undefined)
    .map((row) => ({ split: row.split, percent: row.deteriorating_ratio * 100 }))

  const refresh = () => void analyticsResource.refetch()

  if (analyticsResource.loading && analyticsResource.data === null) {
    return (
      <div className="page-content">
        <PageHeader title="Health Monitoring" subtitle="Fleet health analytics returned by backend evaluation reports." breadcrumb="Monitoring / Health Monitoring" />
        <div className="glass p-12 text-center text-sm text-slate-500">
          <RefreshCw className="mx-auto mb-3 animate-spin text-cyan-400" size={22} />
          Loading health analytics from the backend…
        </div>
      </div>
    )
  }

  if (analyticsResource.error && analyticsResource.data === null) {
    return (
      <div className="page-content">
        <PageHeader title="Health Monitoring" subtitle="Fleet health analytics returned by backend evaluation reports." breadcrumb="Monitoring / Health Monitoring" />
        <div className="glass p-8 text-center">
          <AlertTriangle className="mx-auto mb-3 text-red-400" size={24} />
          <p className="text-sm text-red-300">{analyticsResource.error.message}</p>
          <button type="button" onClick={refresh} className="mt-4 inline-flex items-center gap-2 rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-3 py-2 text-xs font-semibold text-cyan-300"><RefreshCw size={13} /> Retry</button>
        </div>
      </div>
    )
  }

  return (
    <div className="page-content">
      <PageHeader
        title="Health Monitoring"
        subtitle="Fleet health distribution and split-level evaluation metrics loaded from backend report artifacts."
        breadcrumb="Monitoring / Health Monitoring"
        actions={
          <button type="button" onClick={refresh} disabled={analyticsResource.loading} className="inline-flex items-center gap-2 rounded-lg border border-[rgba(30,60,100,0.6)] bg-[rgba(30,60,100,0.4)] px-3 py-2 text-xs font-semibold text-slate-300 disabled:opacity-50">
            <RefreshCw size={13} className={analyticsResource.loading ? 'animate-spin' : ''} /> Refresh
          </button>
        }
      />

      {analyticsResource.error && (
        <div className="mb-5 rounded-lg border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-xs text-amber-300">The last refresh failed: {analyticsResource.error.message}</div>
      )}

      {!dashboardSummary && !healthEvaluation ? (
        <EmptyState message="No dashboard or health evaluation reports were returned by the backend." />
      ) : (
        <>
          <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-5">
            <KpiCard label="Average Health" value={formatNumber(numberAt(healthEvaluation, 'average_health_index'))} sub="Health evaluation report" color="#22c55e" icon={<Activity size={16} />} />
            <KpiCard label="Trend Smoothness" value={formatPercent(numberAt(healthEvaluation, 'average_health_trend_smoothness'))} sub="Evaluation average" color="#06b6d4" icon={<Waves size={16} />} />
            <KpiCard label="Deterioration Consistency" value={formatPercent(numberAt(healthEvaluation, 'average_health_deterioration_consistency'))} sub="Evaluation average" color="#3b82f6" icon={<ShieldCheck size={16} />} />
            <KpiCard label="Test vs Dev Health Drop" value={formatNumber(numberAt(healthEvaluation, 'health_drop_test_vs_dev'))} sub="Reported index difference" color="#f59e0b" icon={<TrendingDown size={16} />} />
            <KpiCard label="Dashboard Average" value={formatNumber(numberAt(dashboardAverages, 'average_health_index'))} sub="Generated dashboard records" color="#8b5cf6" icon={<Activity size={16} />} />
          </div>

          <div className="mb-6 grid grid-cols-1 gap-4 xl:grid-cols-3">
            <div className="glass p-5">
              <SectionHeader title="Health State Distribution" sub="Dashboard summary report" />
              {stateDistribution.length === 0 ? <EmptyState message="No health-state counts were reported." /> : (
                <>
                  <ResponsiveContainer width="100%" height={220}>
                    <PieChart>
                      <Pie data={stateDistribution} dataKey="count" nameKey="state" cx="50%" cy="50%" innerRadius={58} outerRadius={88} paddingAngle={3} strokeWidth={0}>
                        {stateDistribution.map((entry) => <Cell key={entry.state} fill={entry.color} />)}
                      </Pie>
                      <Tooltip contentStyle={chartTooltipStyle} />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="space-y-2">
                    {stateDistribution.map((entry) => (
                      <div key={entry.state} className="flex items-center gap-2 text-xs">
                        <span className="h-2 w-2 rounded-full" style={{ background: entry.color }} />
                        <span className="text-slate-500">{entry.state}</span>
                        <span className="ml-auto font-mono text-slate-300">{formatCount(entry.count)}</span>
                        <span className="w-14 text-right font-mono text-slate-600">{totalStateRecords > 0 ? formatPercent(entry.count / totalStateRecords) : '—'}</span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>

            <div className="glass p-5">
              <SectionHeader title="Average Health by Split" sub="Health evaluation report" />
              {averageBySplit.length === 0 ? <EmptyState message="No split-level health averages were reported." /> : (
                <ResponsiveContainer width="100%" height={270}>
                  <BarChart data={averageBySplit} margin={{ top: 12, right: 10, left: 0, bottom: 0 }}>
                    <XAxis dataKey="split" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={chartTooltipStyle} />
                    <Bar dataKey="average_health_index" name="Average health index" radius={[4, 4, 0, 0]}>
                      {averageBySplit.map((entry) => <Cell key={entry.split} fill={entry.color} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>

            <div className="glass p-5">
              <SectionHeader title="Deteriorating Trend Ratio" sub="Health evaluation report" />
              {deteriorationBySplit.length === 0 ? <EmptyState message="No deteriorating-trend ratios were reported." /> : (
                <ResponsiveContainer width="100%" height={270}>
                  <BarChart data={deteriorationBySplit} margin={{ top: 12, right: 10, left: 0, bottom: 0 }}>
                    <XAxis dataKey="split" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(value) => `${Number(value).toFixed(0)}%`} />
                    <Tooltip contentStyle={chartTooltipStyle} />
                    <Bar dataKey="percent" name="Deteriorating trend" fill="#ef4444" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>

          <div className="glass p-5">
            <SectionHeader title="Split Health Evaluation" sub="Metrics returned by evaluate_health_summary.json" />
            {splitRows.length === 0 ? <EmptyState message="No split-level health evaluation was reported." /> : (
              <div className="overflow-x-auto">
                <table className="data-table w-full">
                  <thead>
                    <tr><th>Split</th><th>Records</th><th>Avg Health</th><th>Healthy</th><th>Degrading</th><th>Warning</th><th>Critical</th><th>Smoothness</th><th>Deterioration Consistency</th></tr>
                  </thead>
                  <tbody>
                    {splitRows.map((row) => (
                      <tr key={row.split}>
                        <td><span className="font-mono font-semibold uppercase text-slate-200">{row.split}</span></td>
                        <td><span className="font-mono text-slate-400">{formatCount(row.row_count)}</span></td>
                        <td><span className="font-mono font-semibold" style={{ color: healthColor(row.average_health_index) }}>{formatNumber(row.average_health_index)}</span></td>
                        <td><span className="font-mono text-green-400">{formatPercent(row.healthy_ratio)}</span></td>
                        <td><span className="font-mono text-blue-400">{formatPercent(row.degrading_ratio)}</span></td>
                        <td><span className="font-mono text-amber-400">{formatPercent(row.warning_ratio)}</span></td>
                        <td><span className="font-mono text-red-400">{formatPercent(row.critical_ratio)}</span></td>
                        <td><span className="font-mono text-slate-400">{formatPercent(row.health_trend_smoothness)}</span></td>
                        <td><span className="font-mono text-slate-400">{formatPercent(row.deterioration_consistency)}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
