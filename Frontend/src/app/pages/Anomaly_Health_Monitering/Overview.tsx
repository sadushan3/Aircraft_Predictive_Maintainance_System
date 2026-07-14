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
  CheckCircle2,
  Database,
  Gauge,
  RefreshCw,
  ShieldAlert,
  Split,
  TrendingDown,
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

const ALERT_COLORS: Record<string, string> = {
  Normal: '#22c55e',
  Watch: '#3b82f6',
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

function stringAt(record: JsonRecord | null, key: string): string | undefined {
  const value = record?.[key]
  return typeof value === 'string' ? value : undefined
}

function formatCount(value: number | undefined): string {
  if (value === undefined) return '—'
  return new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value)
}

function formatDecimal(value: number | undefined, digits = 1): string {
  return value === undefined ? '—' : value.toFixed(digits)
}

function formatPercent(value: number | undefined, digits = 1): string {
  return value === undefined ? '—' : `${(value * 100).toFixed(digits)}%`
}

export default function Overview() {
  const analyticsResource = useDashboardAnalytics()
  const analytics = asRecord(analyticsResource.data)
  const summaries = asRecord(analytics?.summaries)
  const dashboard = asRecord(summaries?.dashboard_data_summary)
  const alertCounts = asRecord(dashboard?.alert_counts)
  const healthCounts = asRecord(dashboard?.health_state_counts)
  const anomalySummary = asRecord(dashboard?.anomaly_summary)
  const averages = asRecord(dashboard?.averages)
  const splitCounts = asRecord(dashboard?.split_counts)
  const sources = Array.isArray(analytics?.sources)
    ? analytics.sources.map(asRecord).filter((source): source is JsonRecord => source !== null)
    : []

  const healthDistribution = useMemo(() => [
    { state: 'Healthy', count: numberAt(healthCounts, 'Healthy'), color: HEALTH_COLORS.Healthy },
    { state: 'Degrading', count: numberAt(healthCounts, 'Degrading'), color: HEALTH_COLORS.Degrading },
    { state: 'Warning', count: numberAt(healthCounts, 'Warning'), color: HEALTH_COLORS.Warning },
    { state: 'Critical', count: numberAt(healthCounts, 'Critical'), color: HEALTH_COLORS.Critical },
  ].filter((item): item is { state: string; count: number; color: string } => item.count !== undefined), [healthCounts])

  const alertDistribution = useMemo(() => [
    { level: 'Normal', count: numberAt(alertCounts, 'Normal'), color: ALERT_COLORS.Normal },
    { level: 'Watch', count: numberAt(alertCounts, 'Watch'), color: ALERT_COLORS.Watch },
    { level: 'Warning', count: numberAt(alertCounts, 'Warning'), color: ALERT_COLORS.Warning },
    { level: 'Critical', count: numberAt(alertCounts, 'Critical'), color: ALERT_COLORS.Critical },
  ].filter((item): item is { level: string; count: number; color: string } => item.count !== undefined), [alertCounts])

  const splitDistribution = useMemo(() => Object.entries(splitCounts ?? {})
    .filter((entry): entry is [string, number] => typeof entry[1] === 'number' && Number.isFinite(entry[1]))
    .map(([name, count]) => ({ name, count })), [splitCounts])

  const refresh = () => void analyticsResource.refetch()

  if (analyticsResource.loading && analyticsResource.data === null) {
    return (
      <div className="page-content">
        <PageHeader title="System Dashboard" subtitle="Fleet analytics loaded from backend report artifacts." breadcrumb="Fleet Overview" />
        <div className="glass p-12 text-center text-sm text-slate-500">
          <RefreshCw className="mx-auto mb-3 animate-spin text-cyan-400" size={22} />
          Loading analytics from the backend…
        </div>
      </div>
    )
  }

  if (analyticsResource.error && analyticsResource.data === null) {
    return (
      <div className="page-content">
        <PageHeader title="System Dashboard" subtitle="Fleet analytics loaded from backend report artifacts." breadcrumb="Fleet Overview" />
        <div className="glass p-8 text-center">
          <AlertTriangle className="mx-auto mb-3 text-red-400" size={24} />
          <p className="text-sm text-red-300">{analyticsResource.error.message}</p>
          <button type="button" onClick={refresh} className="mt-4 inline-flex items-center gap-2 rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-3 py-2 text-xs font-semibold text-cyan-300">
            <RefreshCw size={13} /> Retry
          </button>
        </div>
      </div>
    )
  }

  const totalRecords = numberAt(dashboard, 'records_count')
  const anomalyRecords = numberAt(anomalySummary, 'anomaly_records')
  const anomalyRatio = numberAt(anomalySummary, 'anomaly_ratio')
  const ratioColor = anomalyRatio === undefined
    ? '#64748b'
    : anomalyRatio > 0.3
      ? '#ef4444'
      : anomalyRatio > 0.15
        ? '#f59e0b'
        : '#22c55e'

  return (
    <div className="page-content">
      <PageHeader
        title="System Dashboard"
        subtitle="Fleet analytics loaded from backend report artifacts without scanning the full dashboard dataset."
        breadcrumb="Fleet Overview"
        actions={
          <button type="button" onClick={refresh} disabled={analyticsResource.loading} className="inline-flex items-center gap-2 rounded-lg border border-[rgba(30,60,100,0.6)] bg-[rgba(30,60,100,0.4)] px-3 py-2 text-xs font-semibold text-slate-300 disabled:opacity-50">
            <RefreshCw size={13} className={analyticsResource.loading ? 'animate-spin' : ''} /> Refresh
          </button>
        }
      />

      {analyticsResource.error && (
        <div className="mb-5 rounded-lg border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-xs text-amber-300">
          The last refresh failed: {analyticsResource.error.message}
        </div>
      )}

      {!dashboard ? (
        <EmptyState message="The backend did not return a dashboard summary report." />
      ) : (
        <>
          <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <KpiCard label="Total Records" value={formatCount(totalRecords)} sub="Generated dashboard records" color="#06b6d4" icon={<Database size={16} />} />
            <KpiCard label="Development Records" value={formatCount(numberAt(splitCounts, 'dev'))} sub="Development split" color="#3b82f6" icon={<Split size={16} />} />
            <KpiCard label="Test Records" value={formatCount(numberAt(splitCounts, 'test'))} sub="Test split" color="#8b5cf6" icon={<Split size={16} />} />
            <KpiCard label="Avg Health Index" value={formatDecimal(numberAt(averages, 'average_health_index'))} sub="Reported dashboard average" color="#22c55e" icon={<Activity size={16} />} />
            <KpiCard label="Anomaly Records" value={formatCount(anomalyRecords)} sub="Reported non-normal alerts" color="#f59e0b" icon={<AlertTriangle size={16} />} />
            <KpiCard label="Critical Records" value={formatCount(numberAt(alertCounts, 'Critical'))} sub="Reported critical alerts" color="#ef4444" icon={<ShieldAlert size={16} />} />
            <KpiCard label="Avg Confidence" value={formatPercent(numberAt(averages, 'average_confidence_score'))} sub="Reported confidence score" color="#06b6d4" icon={<Gauge size={16} />} />
            <KpiCard label="Avg Reliability" value={formatPercent(numberAt(averages, 'average_reliability_score'))} sub={`Uncertainty ${formatPercent(numberAt(averages, 'average_uncertainty_score'))}`} color="#22c55e" icon={<CheckCircle2 size={16} />} />
          </div>

          <div className="mb-6 grid grid-cols-1 gap-4 xl:grid-cols-3">
            <div className="glass p-5">
              <SectionHeader title="Fleet Health Distribution" sub="Dashboard summary report" />
              {healthDistribution.length === 0 ? <EmptyState message="No health-state counts were reported." /> : (
                <>
                  <ResponsiveContainer width="100%" height={220}>
                    <PieChart>
                      <Pie data={healthDistribution} dataKey="count" nameKey="state" cx="50%" cy="50%" innerRadius={58} outerRadius={88} paddingAngle={3} strokeWidth={0}>
                        {healthDistribution.map((entry) => <Cell key={entry.state} fill={entry.color} />)}
                      </Pie>
                      <Tooltip contentStyle={chartTooltipStyle} />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="grid grid-cols-2 gap-2">
                    {healthDistribution.map((entry) => (
                      <div key={entry.state} className="flex items-center gap-2 text-xs">
                        <span className="h-2 w-2 rounded-full" style={{ background: entry.color }} />
                        <span className="text-slate-500">{entry.state}</span>
                        <span className="ml-auto font-mono text-slate-300">{formatCount(entry.count)}</span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>

            <div className="glass p-5">
              <SectionHeader title="Alert Distribution" sub="Dashboard summary report" />
              {alertDistribution.length === 0 ? <EmptyState message="No alert counts were reported." /> : (
                <>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={alertDistribution} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                      <XAxis dataKey="level" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(value) => formatCount(Number(value))} />
                      <Tooltip contentStyle={chartTooltipStyle} />
                      <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                        {alertDistribution.map((entry) => <Cell key={entry.level} fill={entry.color} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <div className="flex flex-wrap gap-3">
                    {alertDistribution.map((entry) => (
                      <div key={entry.level} className="flex items-center gap-2 text-xs text-slate-500">
                        <span className="h-2 w-2 rounded-full" style={{ background: entry.color }} />
                        {entry.level}: <span className="font-mono text-slate-300">{formatCount(entry.count)}</span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>

            <div className="glass p-5">
              <SectionHeader title="Anomaly Ratio" sub="Dashboard summary report" />
              {anomalyRatio === undefined ? <EmptyState message="No anomaly ratio was reported." /> : (
                <div className="flex h-[250px] flex-col items-center justify-center">
                  <svg viewBox="0 0 112 112" className="h-36 w-36">
                    <circle cx="56" cy="56" r="46" fill="none" stroke="rgba(30,60,100,0.4)" strokeWidth="10" />
                    <circle cx="56" cy="56" r="46" fill="none" stroke={ratioColor} strokeWidth="10" strokeLinecap="round" strokeDasharray={`${Math.min(Math.max(anomalyRatio, 0), 1) * 289} 289`} transform="rotate(-90 56 56)" />
                    <text x="56" y="54" textAnchor="middle" fill="white" fontSize="17" fontWeight="700">{formatPercent(anomalyRatio)}</text>
                    <text x="56" y="68" textAnchor="middle" fill="#64748b" fontSize="7">ANOMALY RECORDS</text>
                  </svg>
                  <p className="mt-3 text-center text-xs text-slate-500">{formatCount(anomalyRecords)} out of {formatCount(totalRecords)} records</p>
                </div>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <div className="glass p-5">
              <SectionHeader title="Dataset Splits" sub="Counts reported by the dashboard generator" />
              {splitDistribution.length === 0 ? <EmptyState message="No split counts were reported." /> : (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={splitDistribution} layout="vertical" margin={{ top: 8, right: 16, left: 8, bottom: 0 }}>
                    <XAxis type="number" tick={{ fill: '#64748b', fontSize: 10 }} tickFormatter={(value) => formatCount(Number(value))} axisLine={false} tickLine={false} />
                    <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={chartTooltipStyle} />
                    <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>

            <div className="glass p-5">
              <SectionHeader title="Analytics Sources" sub="Report artifacts returned by the backend" />
              {sources.length === 0 ? <EmptyState message="No source metadata was returned." /> : (
                <div className="max-h-64 overflow-auto">
                  <table className="data-table w-full">
                    <thead><tr><th>Report</th><th>Updated</th><th>Size</th></tr></thead>
                    <tbody>
                      {sources.map((source, index) => {
                        const name = stringAt(source, 'name')
                        const updated = stringAt(source, 'updated_at')
                        const size = numberAt(source, 'size_bytes')
                        return (
                          <tr key={name ?? index}>
                            <td><span className="font-mono text-slate-300">{name ?? '—'}</span></td>
                            <td><span className="text-xs text-slate-500">{updated ? new Date(updated).toLocaleString() : '—'}</span></td>
                            <td><span className="font-mono text-slate-400">{formatCount(size)} B</span></td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
