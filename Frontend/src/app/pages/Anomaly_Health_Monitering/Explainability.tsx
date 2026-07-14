import { useMemo, useState } from 'react'
import { BrainCircuit, Database, Layers3, ListOrdered } from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { useExplainability } from '../../../hooks/Anomaly_Health_Monitering'
import { asArray, asRecord, asString, reportContent } from '../../../utils/Anomaly_Health_Monitering/backendData'
import { formatCount, formatMetric, humanize } from '../../../utils/Anomaly_Health_Monitering/presentation'
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

const COLORS = ['#06b6d4', '#3b82f6', '#a855f7', '#f59e0b', '#22c55e', '#f97316']

export default function Explainability() {
  const explainability = useExplainability()
  const [selectedModel, setSelectedModel] = useState<string>('')

  const payload = asRecord(explainability.data as unknown)
  const reports = asArray(payload.reports)
  const rows = asArray(payload.shap_rows).map(asRecord)
  const shapSummary = reportContent(reports, 'shap_summary.json')
  const subsystemSummary = reportContent(reports, 'subsystem_explanations_summary.json')
  const rankingSummary = reportContent(reports, 'sensor_residual_ranking_summary.json')

  const models = useMemo(
    () => Array.from(new Set(rows.map((row) => asString(row.model)).filter((value): value is string => Boolean(value)))),
    [rows],
  )
  const activeModel = selectedModel || models[0] || ''

  const modelRows = rows
    .filter((row) => asString(row.model) === activeModel)
    .map((row) => ({
      feature: asString(row.feature) ?? 'Unknown',
      value: Number(row.mean_abs_shap),
      type: asString(row.explanation_type) ?? 'Not reported',
    }))
    .filter((row) => Number.isFinite(row.value))
    .sort((left, right) => right.value - left.value)

  const globalFeatures = asArray(shapSummary.top_features)
    .map(asRecord)
    .map((row) => ({ feature: asString(row.feature) ?? 'Unknown', value: Number(row.mean_abs_shap) }))
    .filter((row) => Number.isFinite(row.value))

  const subsystemCounts = Object.entries(asRecord(subsystemSummary.primary_subsystem_counts))
    .map(([name, count]) => ({ name: humanize(name), count: Number(count) }))
    .filter((item) => Number.isFinite(item.count))
    .sort((left, right) => right.count - left.count)

  const residualCounts = Object.entries(asRecord(rankingSummary.top_1_sensor_counts))
    .map(([sensor, count]) => ({ sensor, count: Number(count) }))
    .filter((item) => Number.isFinite(item.count))
    .sort((left, right) => right.count - left.count)
    .slice(0, 8)

  if (explainability.loading) return <div className="page-content"><LoadingState message="Loading explainability artifacts…" /></div>
  if (explainability.error) {
    return <div className="page-content"><ErrorState error={explainability.error.message} onRetry={() => void explainability.refetch()} /></div>
  }
  if (reports.length === 0 && rows.length === 0) {
    return <div className="page-content"><EmptyState message="No explainability artifacts are available. Run the explainability stage first." /></div>
  }

  return (
    <div className="page-content">
      <PageHeader
        title="Explainability"
        subtitle="Model attribution, subsystem mapping, and residual rankings loaded from backend-generated artifacts."
        breadcrumb="Intelligence / Explainability"
      />

      <div className="grid grid-cols-4 gap-4 mb-6">
        <KpiCard label="SHAP sample size" value={formatCount(Number(shapSummary.sample_size))} color="#06b6d4" icon={<Database size={16} />} />
        <KpiCard label="Explained features" value={formatCount(Number(shapSummary.feature_count))} color="#3b82f6" icon={<BrainCircuit size={16} />} />
        <KpiCard label="Models explained" value={formatCount(models.length)} color="#a855f7" icon={<Layers3 size={16} />} />
        <KpiCard label="Residual sensors" value={formatCount(Number(rankingSummary.abs_residual_sensor_count))} color="#f59e0b" icon={<ListOrdered size={16} />} />
      </div>

      <div className="grid grid-cols-2 gap-5 mb-5">
        <div className="glass p-5">
          <SectionHeader title="Global Feature Importance" sub="Mean absolute SHAP across explained models" />
          {globalFeatures.length === 0 ? <EmptyState /> : (
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={globalFeatures.slice(0, 12)} layout="vertical" margin={{ left: 25, right: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,60,100,0.25)" />
                <XAxis type="number" tick={{ fill: '#64748b', fontSize: 10 }} />
                <YAxis dataKey="feature" type="category" width={85} tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <Tooltip contentStyle={chartTooltipStyle} formatter={(value) => formatMetric(Number(value), 5)} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {globalFeatures.slice(0, 12).map((item, index) => <Cell key={item.feature} fill={COLORS[index % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="glass p-5">
          <SectionHeader
            title="Per-Model SHAP"
            sub="Persisted bounded SHAP output"
            action={models.length > 0 ? (
              <select
                value={activeModel}
                onChange={(event) => setSelectedModel(event.target.value)}
                className="input-field text-xs"
              >
                {models.map((model) => <option key={model} value={model}>{humanize(model)}</option>)}
              </select>
            ) : undefined}
          />
          {modelRows.length === 0 ? <EmptyState /> : (
            <>
              <ResponsiveContainer width="100%" height={310}>
                <BarChart data={modelRows.slice(0, 12)} layout="vertical" margin={{ left: 25, right: 25 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,60,100,0.25)" />
                  <XAxis type="number" tick={{ fill: '#64748b', fontSize: 10 }} />
                  <YAxis dataKey="feature" type="category" width={85} tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <Tooltip contentStyle={chartTooltipStyle} formatter={(value) => formatMetric(Number(value), 5)} />
                  <Bar dataKey="value" fill="#a855f7" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <div className="text-xs text-slate-600 mt-2">Method: {modelRows[0]?.type}</div>
            </>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-5">
        <div className="glass p-5">
          <SectionHeader title="Subsystem Attribution" sub="Primary subsystem counts from generated explanations" />
          {subsystemCounts.length === 0 ? <EmptyState /> : (
            <div className="space-y-3">
              {subsystemCounts.map((item, index) => {
                const maximum = subsystemCounts[0]?.count || 1
                return (
                  <div key={item.name}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-400">{item.name}</span>
                      <span className="font-mono" style={{ color: COLORS[index % COLORS.length] }}>{formatCount(item.count)}</span>
                    </div>
                    <div className="h-2 bg-[rgba(30,60,100,0.35)] rounded-full overflow-hidden">
                      <div className="h-full rounded-full" style={{ width: `${(item.count / maximum) * 100}%`, background: COLORS[index % COLORS.length] }} />
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        <div className="glass p-5">
          <SectionHeader title="Top Residual Sensor Ranking" sub="Rank-one sensor frequency from residual analysis" />
          {residualCounts.length === 0 ? <EmptyState /> : (
            <div className="space-y-2">
              {residualCounts.map((item, index) => (
                <div key={item.sensor} className="flex items-center gap-3 py-2 border-b border-[rgba(30,60,100,0.25)] last:border-0">
                  <span className="font-mono text-xs text-slate-600 w-5">#{index + 1}</span>
                  <span className="font-mono text-sm text-slate-300 flex-1">{item.sensor}</span>
                  <span className="font-mono text-xs text-cyan-400">{formatCount(item.count)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <Disclaimer text="SHAP values and residual rankings explain model behaviour and support inspection. They do not prove physical causality." />
    </div>
  )
}
