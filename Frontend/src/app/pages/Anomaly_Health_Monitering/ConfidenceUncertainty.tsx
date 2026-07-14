import { useState } from 'react'
import { Gauge, Search, ShieldCheck, ShieldQuestion, Waves } from 'lucide-react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { useConfidence, useDashboardAnalytics } from '../../../hooks/Anomaly_Health_Monitering'
import { asRecord, recordEntries } from '../../../utils/Anomaly_Health_Monitering/backendData'
import { formatCount, formatPercent, humanize } from '../../../utils/Anomaly_Health_Monitering/presentation'
import {
  Btn,
  CircularGauge,
  Disclaimer,
  EmptyState,
  ErrorState,
  KpiCard,
  LoadingState,
  PageHeader,
  ScoreBar,
  SectionHeader,
  chartTooltipStyle,
} from '../../components/ui/Anomaly_Health_Monitering/ui'

export default function ConfidenceUncertainty() {
  const analytics = useDashboardAnalytics()
  const [unitInput, setUnitInput] = useState('')
  const [selectedUnit, setSelectedUnit] = useState<number | null>(null)
  const confidenceTrend = useConfidence(selectedUnit)

  const summaries = asRecord(analytics.data?.summaries)
  const confidence = asRecord(summaries.confidence_scores_summary)
  const agreement = asRecord(summaries.model_agreement_summary)
  const agreementScores = asRecord(agreement.score_summary)
  const weights = asRecord(confidence.weights)
  const labels = recordEntries(confidence.confidence_label_counts)
    .map(([name, count]) => ({ name: humanize(name), count: Number(count) }))
    .filter((item) => Number.isFinite(item.count))
  const splitSummary = recordEntries(confidence.split_summary)
    .map(([split, value]) => ({ split, values: asRecord(value) }))

  const avgConfidence = Number(confidence.average_confidence_score)
  const avgUncertainty = Number(confidence.average_uncertainty_score)
  const avgReliability = Number(confidence.average_reliability_score)
  const avgAgreement = Number(agreementScores.average_model_agreement_score)

  const trend = (confidenceTrend.data ?? [])
    .filter((row) => row.cycle != null)
    .map((row) => ({
      cycle: Number(row.cycle),
      confidence: row.confidence_score == null ? null : Number(row.confidence_score),
      uncertainty: row.uncertainty_score == null ? null : Number(row.uncertainty_score),
      reliability: row.reliability_score == null ? null : Number(row.reliability_score),
      agreement: row.model_agreement_score == null ? null : Number(row.model_agreement_score),
    }))

  function loadUnit() {
    const parsed = Number(unitInput)
    if (Number.isInteger(parsed) && parsed >= 0) setSelectedUnit(parsed)
  }

  if (analytics.loading) return <div className="page-content"><LoadingState message="Loading confidence summaries…" /></div>
  if (analytics.error) {
    return <div className="page-content"><ErrorState error={analytics.error.message} onRetry={() => void analytics.refetch()} /></div>
  }

  return (
    <div className="page-content">
      <PageHeader
        title="Confidence & Uncertainty"
        subtitle="Confidence, uncertainty, reliability, and model agreement calculated by the backend ensemble."
        breadcrumb="Intelligence / Confidence & Uncertainty"
      />

      <div className="grid grid-cols-4 gap-4 mb-6">
        <KpiCard label="Avg confidence" value={formatPercent(Number.isFinite(avgConfidence) ? avgConfidence : null)} color="#06b6d4" icon={<ShieldCheck size={16} />} />
        <KpiCard label="Avg uncertainty" value={formatPercent(Number.isFinite(avgUncertainty) ? avgUncertainty : null)} color="#a855f7" icon={<ShieldQuestion size={16} />} />
        <KpiCard label="Avg reliability" value={formatPercent(Number.isFinite(avgReliability) ? avgReliability : null)} color="#22c55e" icon={<Gauge size={16} />} />
        <KpiCard label="Model agreement" value={formatPercent(Number.isFinite(avgAgreement) ? avgAgreement : null)} color="#f59e0b" icon={<Waves size={16} />} />
      </div>

      <div className="grid grid-cols-3 gap-5 mb-5">
        <div className="glass p-5 col-span-2">
          <SectionHeader title="Confidence Construction" sub="Weights loaded from confidence_scores_summary.json" />
          {Object.keys(weights).length === 0 ? <EmptyState /> : (
            <div className="grid grid-cols-2 gap-x-8">
              {recordEntries(weights).map(([name, value], index) => {
                const numberValue = Number(value)
                return Number.isFinite(numberValue) ? (
                  <ScoreBar
                    key={name}
                    label={humanize(name)}
                    value={numberValue}
                    color={['#06b6d4', '#3b82f6', '#a855f7', '#22c55e'][index % 4]}
                  />
                ) : null
              })}
            </div>
          )}
        </div>
        <div className="glass p-5 flex justify-around items-center">
          {Number.isFinite(avgConfidence) && <CircularGauge value={avgConfidence} max={1} size={120} color="#06b6d4" label="Confidence" sub="fleet avg" />}
          {Number.isFinite(avgUncertainty) && <CircularGauge value={avgUncertainty} max={1} size={120} color="#a855f7" label="Uncertainty" sub="fleet avg" />}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-5 mb-5">
        <div className="glass p-5">
          <SectionHeader title="Confidence Labels" sub="All scored records" />
          {labels.length === 0 ? <EmptyState /> : (
            <div className="space-y-3">
              {labels.map((item, index) => (
                <div key={item.name} className="flex items-center justify-between py-2 border-b border-[rgba(30,60,100,0.25)] last:border-0">
                  <span className="text-sm text-slate-400">{item.name}</span>
                  <span className="font-mono text-sm" style={{ color: ['#22c55e', '#06b6d4', '#f59e0b', '#ef4444'][index % 4] }}>
                    {formatCount(item.count)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="glass p-5">
          <SectionHeader title="Split Comparison" sub="Backend confidence summary by data split" />
          {splitSummary.length === 0 ? <EmptyState /> : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead><tr className="text-left text-[10px] uppercase tracking-wider text-slate-600 border-b border-[rgba(30,60,100,0.3)]">
                  <th className="py-2">Split</th><th>Rows</th><th>Confidence</th><th>Uncertainty</th><th>Reliability</th>
                </tr></thead>
                <tbody>{splitSummary.map((row) => (
                  <tr key={row.split} className="border-b border-[rgba(30,60,100,0.2)] last:border-0">
                    <td className="py-3 font-mono text-cyan-400">{row.split}</td>
                    <td className="font-mono text-slate-400">{formatCount(Number(row.values.rows))}</td>
                    <td className="font-mono text-slate-300">{formatPercent(Number(row.values.average_confidence_score))}</td>
                    <td className="font-mono text-slate-300">{formatPercent(Number(row.values.average_uncertainty_score))}</td>
                    <td className="font-mono text-slate-300">{formatPercent(Number(row.values.average_reliability_score))}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      <div className="glass p-5">
        <SectionHeader
          title="Unit Confidence Trend"
          sub="Enter a unit ID to query its persisted confidence records"
          action={(
            <div className="flex gap-2">
              <input
                value={unitInput}
                onChange={(event) => setUnitInput(event.target.value)}
                onKeyDown={(event) => { if (event.key === 'Enter') loadUnit() }}
                inputMode="numeric"
                placeholder="Unit ID"
                className="input-field w-28 text-xs"
              />
              <Btn size="sm" onClick={loadUnit} disabled={!unitInput.trim()}><Search size={12} /> Load</Btn>
            </div>
          )}
        />
        {selectedUnit == null ? <EmptyState message="Enter a unit ID to load its real confidence trend." /> : confidenceTrend.loading ? (
          <LoadingState message={`Loading unit ${selectedUnit}…`} />
        ) : confidenceTrend.error ? (
          <ErrorState error={confidenceTrend.error.message} onRetry={() => void confidenceTrend.refetch()} />
        ) : trend.length === 0 ? (
          <EmptyState message={`No confidence records were returned for unit ${selectedUnit}.`} />
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trend}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,60,100,0.25)" />
              <XAxis dataKey="cycle" tick={{ fill: '#64748b', fontSize: 10 }} />
              <YAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 10 }} />
              <Tooltip contentStyle={chartTooltipStyle} />
              <Line type="monotone" dataKey="confidence" stroke="#06b6d4" dot={false} strokeWidth={2} connectNulls />
              <Line type="monotone" dataKey="uncertainty" stroke="#a855f7" dot={false} strokeWidth={2} connectNulls />
              <Line type="monotone" dataKey="reliability" stroke="#22c55e" dot={false} strokeWidth={2} connectNulls />
              <Line type="monotone" dataKey="agreement" stroke="#f59e0b" dot={false} strokeWidth={2} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      <Disclaimer text="Confidence scores express model-output reliability. They do not guarantee physical correctness or prescribe maintenance action." />
    </div>
  )
}
