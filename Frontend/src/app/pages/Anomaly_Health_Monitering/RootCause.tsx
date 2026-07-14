import { BarChart3, GitBranch, Network, Radar, Waves } from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { useReasoning } from '../../../hooks/Anomaly_Health_Monitering'
import { asArray, asRecord, recordEntries, reportContent } from '../../../utils/Anomaly_Health_Monitering/backendData'
import { formatCount, humanize } from '../../../utils/Anomaly_Health_Monitering/presentation'
import {
  Disclaimer,
  EmptyState,
  ErrorState,
  KpiCard,
  LoadingState,
  PageHeader,
  SectionHeader,
  chartTooltipStyle,
} from '../../components/ui/Anomaly_Health_Monitering/ui'

const COLORS = ['#06b6d4', '#3b82f6', '#a855f7', '#f59e0b', '#22c55e', '#f97316', '#ef4444']

function countSeries(value: unknown, limit?: number) {
  return recordEntries(value)
    .map(([name, count]) => ({ name: humanize(name), count: Number(count) }))
    .filter((item) => Number.isFinite(item.count))
    .sort((left, right) => right.count - left.count)
    .slice(0, limit)
}

export default function RootCause() {
  const reasoning = useReasoning()

  if (reasoning.loading) return <div className="page-content"><LoadingState message="Loading reasoning reports…" /></div>
  if (reasoning.error) {
    return <div className="page-content"><ErrorState error={reasoning.error.message} onRetry={() => void reasoning.refetch()} /></div>
  }

  const payload = asRecord(reasoning.data as unknown)
  const reports = asArray(payload.reports)
  const rootCause = reportContent(reports, 'root_cause_summary.json')
  const temporal = reportContent(reports, 'temporal_reasoning_summary.json')
  const dependency = reportContent(reports, 'sensor_dependency_graph_summary.json')
  const memory = reportContent(reports, 'root_cause_memory_summary.json')

  const patterns = countSeries(rootCause.pattern_counts)
  const sensors = countSeries(rootCause.top_sensor_1_counts, 10)
  const temporalPatterns = countSeries(temporal.temporal_pattern_counts)
  const families = countSeries(dependency.family_counts)
  const dependencies = countSeries(dependency.dependency_counts)

  if (reports.length === 0) {
    return <div className="page-content"><EmptyState message="No backend reasoning reports are available. Run the reasoning stage first." /></div>
  }

  return (
    <div className="page-content">
      <PageHeader
        title="Root-Cause Reasoning"
        subtitle="Residual patterns, temporal behaviour, and sensor relationships read from generated backend reports."
        breadcrumb="Intelligence / Root-Cause Reasoning"
      />

      <div className="grid grid-cols-4 gap-4 mb-6">
        <KpiCard label="Reasoning records" value={formatCount(Number(rootCause.records_count))} color="#06b6d4" icon={<GitBranch size={16} />} />
        <KpiCard label="Pattern classes" value={formatCount(patterns.length)} color="#3b82f6" icon={<BarChart3 size={16} />} />
        <KpiCard label="Temporal records" value={formatCount(Number(temporal.records_count))} color="#a855f7" icon={<Waves size={16} />} />
        <KpiCard label="Dependency rows" value={formatCount(Number(dependency.records_count))} color="#f59e0b" icon={<Network size={16} />} />
      </div>

      <div className="grid grid-cols-2 gap-5 mb-5">
        <div className="glass p-5">
          <SectionHeader title="Root-Cause Pattern Distribution" sub="All persisted reasoning records" />
          {patterns.length === 0 ? <EmptyState /> : (
            <ResponsiveContainer width="100%" height={330}>
              <BarChart data={patterns} layout="vertical" margin={{ left: 25, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,60,100,0.25)" />
                <XAxis type="number" tick={{ fill: '#64748b', fontSize: 10 }} tickFormatter={(value) => formatCount(Number(value))} />
                <YAxis dataKey="name" type="category" width={190} tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <Tooltip contentStyle={chartTooltipStyle} formatter={(value) => formatCount(Number(value))} />
                <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                  {patterns.map((item, index) => <Cell key={item.name} fill={COLORS[index % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="glass p-5">
          <SectionHeader title="Leading Residual Sensors" sub="Rank-one contributor frequency" />
          {sensors.length === 0 ? <EmptyState /> : (
            <ResponsiveContainer width="100%" height={330}>
              <BarChart data={sensors} margin={{ left: 5, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,60,100,0.25)" />
                <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 10 }} angle={-35} textAnchor="end" height={70} />
                <YAxis tick={{ fill: '#64748b', fontSize: 10 }} tickFormatter={(value) => formatCount(Number(value))} />
                <Tooltip contentStyle={chartTooltipStyle} formatter={(value) => formatCount(Number(value))} />
                <Bar dataKey="count" fill="#06b6d4" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-5">
        <div className="glass p-5 col-span-2">
          <SectionHeader title="Temporal Reasoning Patterns" sub="Labels produced by the temporal reasoning stage" />
          {temporalPatterns.length === 0 ? <EmptyState /> : (
            <ResponsiveContainer width="100%" height={270}>
              <BarChart data={temporalPatterns}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,60,100,0.25)" />
                <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 10 }} angle={-20} textAnchor="end" height={65} />
                <YAxis tick={{ fill: '#64748b', fontSize: 10 }} tickFormatter={(value) => formatCount(Number(value))} />
                <Tooltip contentStyle={chartTooltipStyle} formatter={(value) => formatCount(Number(value))} />
                <Bar dataKey="count" fill="#a855f7" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="glass p-5">
          <SectionHeader title="Sensor Families" sub="Dependency graph membership" />
          {families.length === 0 ? <EmptyState /> : (
            <ResponsiveContainer width="100%" height={190}>
              <PieChart>
                <Pie data={families} dataKey="count" nameKey="name" innerRadius={45} outerRadius={76}>
                  {families.map((item, index) => <Cell key={item.name} fill={COLORS[index % COLORS.length]} />)}
                </Pie>
                <Tooltip contentStyle={chartTooltipStyle} />
              </PieChart>
            </ResponsiveContainer>
          )}
          <div className="space-y-2 mt-2">
            {dependencies.map((item, index) => (
              <div key={item.name} className="flex items-center justify-between gap-3 text-xs">
                <span className="text-slate-500 truncate"><Radar size={11} className="inline mr-1" />{item.name}</span>
                <span className="font-mono" style={{ color: COLORS[index % COLORS.length] }}>{formatCount(item.count)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {Object.keys(memory).length > 0 && (
        <div className="mt-5 text-xs text-slate-600 font-mono">
          Root-cause memory report status: {String(memory.status ?? 'available')}
        </div>
      )}
      <Disclaimer text="Subsystem and root-cause labels support investigation; they are not confirmed physical causality or maintenance decisions." />
    </div>
  )
}
