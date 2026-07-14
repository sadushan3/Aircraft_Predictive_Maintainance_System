import { useMemo, useState, type FormEvent } from 'react'
import { CheckCircle, Info, Send } from 'lucide-react'
import { submitFeedback } from '../../../api/Anomaly_Health_Monitering'
import {
  useApiMutation,
  useFeedback,
  useThresholds,
} from '../../../hooks/Anomaly_Health_Monitering'
import type {
  ApiResponse,
  FeedbackLabel,
  FeedbackRecord,
  FeedbackRequest,
} from '../../../types/Anomaly_Health_Monitering'
import {
  formatMetric,
  getAlertColor,
  humanize,
  toFiniteNumber,
} from '../../../utils/Anomaly_Health_Monitering/presentation'
import {
  Badge,
  Disclaimer,
  EmptyState,
  ErrorState,
  LoadingState,
  PageHeader,
  SectionHeader,
} from '../../components/ui/Anomaly_Health_Monitering/ui'

const FEEDBACK_LABELS: FeedbackLabel[] = [
  'accepted_alert',
  'rejected_false_alarm',
  'missed_anomaly',
  'uncertain',
]

const FEEDBACK_LABEL_COLORS: Record<FeedbackLabel, string> = {
  accepted_alert: '#22c55e',
  rejected_false_alarm: '#3b82f6',
  missed_anomaly: '#ef4444',
  uncertain: '#f59e0b',
}

interface FeedbackHistoryPayload {
  feedback?: FeedbackRecord[]
  recent_alerts?: FeedbackRecord[]
  feedback_total_records?: number
  feedback_truncated?: boolean
}

interface ArtifactMetadata {
  name?: string
  updated_at?: string | null
  size_bytes?: number | null
}

interface ThresholdContent {
  thresholds?: Record<string, Record<string, unknown>>
  adjustment_step?: number | null
  feedback_records_used?: number | null
  rules?: Record<string, unknown>
}

interface ThresholdPayload {
  metadata?: ArtifactMetadata
  content?: ThresholdContent
}

interface FeedbackFormState {
  unit_id: string
  cycle: string
  context_id: string
  alert_level: string
  final_anomaly_score: string
  root_cause_pattern: string
  feedback_label: FeedbackLabel
  operator_note: string
}

const EMPTY_FORM: FeedbackFormState = {
  unit_id: '',
  cycle: '',
  context_id: '',
  alert_level: '',
  final_anomaly_score: '',
  root_cause_pattern: '',
  feedback_label: 'accepted_alert',
  operator_note: '',
}

function displayValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return 'N/A'
  if (typeof value === 'number') return Number.isFinite(value) ? value.toLocaleString() : 'N/A'
  return String(value)
}

function formatTimestamp(value: unknown): string {
  if (!value) return 'N/A'
  const date = new Date(String(value))
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString()
}

function feedbackLabel(record: FeedbackRecord): string {
  return String(record.feedback_label ?? record.feedback_status ?? 'unreviewed')
}

export default function FeedbackLearning() {
  const [form, setForm] = useState<FeedbackFormState>(EMPTY_FORM)
  const [validationError, setValidationError] = useState<string | null>(null)

  const feedbackResource = useFeedback({ keepPreviousData: true })
  const thresholdResource = useThresholds({ keepPreviousData: true })

  // The dashboard API returns both persisted feedback and a bounded alert sample.
  const feedbackPayload = feedbackResource.data as unknown as FeedbackHistoryPayload | null
  const thresholdPayload = thresholdResource.data as unknown as ThresholdPayload | null
  const feedbackRecords = feedbackPayload?.feedback ?? []
  const recentAlerts = feedbackPayload?.recent_alerts ?? []
  const thresholdContent = thresholdPayload?.content
  const thresholdGroups = Object.entries(thresholdContent?.thresholds ?? {})

  const feedbackMutation = useApiMutation<ApiResponse<unknown>, FeedbackRequest>(
    async (request, signal) => submitFeedback(request, { signal }),
  )

  const suggestions = useMemo(() => ({
    units: [...new Set(recentAlerts.map((row) => row.unit_id).filter((value) => value != null))],
    contexts: [...new Set(recentAlerts.map((row) => row.context_id).filter((value) => value != null))],
    alertLevels: [...new Set(recentAlerts.map((row) => row.alert_level).filter((value) => Boolean(value)))],
    patterns: [...new Set(recentAlerts.map((row) => row.root_cause_pattern).filter((value) => Boolean(value)))],
  }), [recentAlerts])

  const feedbackCounts = useMemo(
    () => FEEDBACK_LABELS.map((label) => ({
      label,
      count: feedbackRecords.filter((record) => feedbackLabel(record) === label).length,
    })),
    [feedbackRecords],
  )

  const updateField = <K extends keyof FeedbackFormState>(
    field: K,
    value: FeedbackFormState[K],
  ) => {
    setForm((current) => ({ ...current, [field]: value }))
    setValidationError(null)
    if (feedbackMutation.data || feedbackMutation.error) feedbackMutation.reset()
  }

  const selectAlert = (alert: FeedbackRecord) => {
    setForm((current) => ({
      ...current,
      unit_id: alert.unit_id == null ? '' : String(alert.unit_id),
      cycle: alert.cycle == null ? '' : String(alert.cycle),
      context_id: alert.context_id == null ? '' : String(alert.context_id),
      alert_level: alert.alert_level == null ? '' : String(alert.alert_level),
      final_anomaly_score:
        alert.final_anomaly_score == null ? '' : String(alert.final_anomaly_score),
      root_cause_pattern:
        alert.root_cause_pattern == null ? '' : String(alert.root_cause_pattern),
    }))
    setValidationError(null)
    feedbackMutation.reset()
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    const unitId = Number(form.unit_id)
    const cycle = Number(form.cycle)
    const anomalyScore = form.final_anomaly_score.trim()
      ? Number(form.final_anomaly_score)
      : null

    if (!Number.isInteger(unitId) || unitId < 0 || !Number.isInteger(cycle) || cycle < 0) {
      setValidationError('Unit ID and cycle must be non-negative whole numbers.')
      return
    }
    if (anomalyScore !== null && !Number.isFinite(anomalyScore)) {
      setValidationError('Final anomaly score must be a valid number.')
      return
    }

    const request: FeedbackRequest = {
      unit_id: unitId,
      cycle,
      feedback_label: form.feedback_label,
      context_id: form.context_id.trim() || null,
      alert_level: form.alert_level.trim() || null,
      final_anomaly_score: anomalyScore,
      root_cause_pattern: form.root_cause_pattern.trim() || null,
      operator_note: form.operator_note.trim() || null,
    }

    try {
      await feedbackMutation.mutate(request)
      await Promise.all([feedbackResource.refetch(), thresholdResource.refetch()])
    } catch {
      // The mutation hook exposes the backend error in the form.
    }
  }

  return (
    <div className="page-content">
      <PageHeader
        title="Feedback Learning"
        subtitle="Validate real alerts and review the threshold state persisted by the backend."
        breadcrumb="Intelligence / Feedback Learning"
      />

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className="flex flex-col gap-4">
          <div className="glass p-5">
            <SectionHeader title="Submit Alert Feedback" sub="Operator validation sent to the backend" />

            {feedbackMutation.data && (
              <div className="flex items-start gap-2 p-3 rounded-lg mb-4 bg-green-500/10 border border-green-500/25 text-green-400 text-sm">
                <CheckCircle size={15} className="mt-0.5 flex-shrink-0" />
                <span>{feedbackMutation.data.message}</span>
              </div>
            )}
            {(validationError || feedbackMutation.error) && (
              <div className="p-3 rounded-lg mb-4 bg-red-500/10 border border-red-500/25 text-red-400 text-sm">
                {validationError ?? feedbackMutation.error?.message}
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <label className="text-xs font-mono text-slate-600 uppercase tracking-widest">
                  Unit ID
                  <input
                    required
                    min="0"
                    step="1"
                    type="number"
                    list="feedback-units"
                    value={form.unit_id}
                    onChange={(event) => updateField('unit_id', event.target.value)}
                    className="mt-1.5 w-full px-3 py-2 rounded-lg text-sm font-mono bg-[rgba(30,60,100,0.3)] border border-[rgba(30,60,100,0.5)] text-slate-300 outline-none focus:border-cyan-500/50"
                  />
                  <datalist id="feedback-units">
                    {suggestions.units.map((unit) => <option key={String(unit)} value={String(unit)} />)}
                  </datalist>
                </label>
                <label className="text-xs font-mono text-slate-600 uppercase tracking-widest">
                  Cycle
                  <input
                    required
                    min="0"
                    step="1"
                    type="number"
                    value={form.cycle}
                    onChange={(event) => updateField('cycle', event.target.value)}
                    className="mt-1.5 w-full px-3 py-2 rounded-lg text-sm font-mono bg-[rgba(30,60,100,0.3)] border border-[rgba(30,60,100,0.5)] text-slate-300 outline-none focus:border-cyan-500/50"
                  />
                </label>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <label className="text-xs font-mono text-slate-600 uppercase tracking-widest">
                  Context ID
                  <input
                    list="feedback-contexts"
                    value={form.context_id}
                    onChange={(event) => updateField('context_id', event.target.value)}
                    className="mt-1.5 w-full px-3 py-2 rounded-lg text-sm font-mono bg-[rgba(30,60,100,0.3)] border border-[rgba(30,60,100,0.5)] text-slate-300 outline-none focus:border-cyan-500/50"
                  />
                  <datalist id="feedback-contexts">
                    {suggestions.contexts.map((context) => (
                      <option key={String(context)} value={String(context)} />
                    ))}
                  </datalist>
                </label>
                <label className="text-xs font-mono text-slate-600 uppercase tracking-widest">
                  Alert Level
                  <input
                    list="feedback-alert-levels"
                    value={form.alert_level}
                    onChange={(event) => updateField('alert_level', event.target.value)}
                    className="mt-1.5 w-full px-3 py-2 rounded-lg text-sm font-mono bg-[rgba(30,60,100,0.3)] border border-[rgba(30,60,100,0.5)] text-slate-300 outline-none focus:border-cyan-500/50"
                  />
                  <datalist id="feedback-alert-levels">
                    {suggestions.alertLevels.map((level) => (
                      <option key={String(level)} value={String(level)} />
                    ))}
                  </datalist>
                </label>
              </div>

              <label className="text-xs font-mono text-slate-600 uppercase tracking-widest block">
                Final Anomaly Score
                <input
                  type="number"
                  step="any"
                  value={form.final_anomaly_score}
                  onChange={(event) => updateField('final_anomaly_score', event.target.value)}
                  className="mt-1.5 w-full px-3 py-2 rounded-lg text-sm font-mono bg-[rgba(30,60,100,0.3)] border border-[rgba(30,60,100,0.5)] text-slate-300 outline-none focus:border-cyan-500/50"
                />
              </label>

              <label className="text-xs font-mono text-slate-600 uppercase tracking-widest block">
                Root-Cause Pattern
                <input
                  list="feedback-patterns"
                  value={form.root_cause_pattern}
                  onChange={(event) => updateField('root_cause_pattern', event.target.value)}
                  className="mt-1.5 w-full px-3 py-2 rounded-lg text-sm font-mono bg-[rgba(30,60,100,0.3)] border border-[rgba(30,60,100,0.5)] text-slate-300 outline-none focus:border-cyan-500/50"
                />
                <datalist id="feedback-patterns">
                  {suggestions.patterns.map((pattern) => (
                    <option key={String(pattern)} value={String(pattern)} />
                  ))}
                </datalist>
              </label>

              <label className="text-xs font-mono text-slate-600 uppercase tracking-widest block">
                Feedback Label
                <select
                  value={form.feedback_label}
                  onChange={(event) => updateField('feedback_label', event.target.value as FeedbackLabel)}
                  className="mt-1.5 w-full px-3 py-2 rounded-lg text-sm font-mono bg-[rgba(30,60,100,0.3)] border border-[rgba(30,60,100,0.5)] text-slate-300 outline-none focus:border-cyan-500/50"
                >
                  {FEEDBACK_LABELS.map((label) => (
                    <option key={label} value={label}>{humanize(label)}</option>
                  ))}
                </select>
              </label>

              <label className="text-xs font-mono text-slate-600 uppercase tracking-widest block">
                Operator Note
                <textarea
                  maxLength={2000}
                  value={form.operator_note}
                  onChange={(event) => updateField('operator_note', event.target.value)}
                  rows={3}
                  className="mt-1.5 w-full px-3 py-2 rounded-lg text-sm font-mono bg-[rgba(30,60,100,0.3)] border border-[rgba(30,60,100,0.5)] text-slate-300 outline-none focus:border-cyan-500/50 resize-none"
                />
              </label>

              <button
                type="submit"
                disabled={feedbackMutation.loading}
                className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-semibold transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                style={{ background: 'linear-gradient(135deg, #06b6d4, #3b82f6)', color: 'white' }}
              >
                <Send size={14} />
                {feedbackMutation.loading ? 'Submitting to backend…' : 'Submit Feedback'}
              </button>
            </form>
          </div>

          <div className="glass p-5">
            <div className="text-xs font-mono text-slate-600 uppercase tracking-widest mb-3">
              Persisted Feedback Summary
            </div>
            {feedbackResource.loading && !feedbackPayload ? (
              <LoadingState message="Loading feedback history…" />
            ) : feedbackRecords.length === 0 ? (
              <EmptyState message="No operator feedback has been persisted yet." />
            ) : (
              feedbackCounts.map(({ label, count }) => (
                <div key={label} className="flex items-center gap-2 py-2 border-b border-[rgba(30,60,100,0.2)] last:border-0">
                  <span className="w-2 h-2 rounded-full" style={{ background: FEEDBACK_LABEL_COLORS[label] }} />
                  <span className="text-xs text-slate-400">{humanize(label)}</span>
                  <span className="ml-auto font-mono text-xs" style={{ color: FEEDBACK_LABEL_COLORS[label] }}>
                    {count.toLocaleString()}
                  </span>
                </div>
              ))
            )}
            {feedbackPayload?.feedback_truncated && (
              <div className="mt-3 text-[10px] text-slate-600 font-mono">
                Showing the newest {feedbackRecords.length.toLocaleString()} of{' '}
                {displayValue(feedbackPayload.feedback_total_records)} records.
              </div>
            )}
          </div>
        </div>

        <div className="xl:col-span-2 flex flex-col gap-4">
          <div className="glass p-5">
            <SectionHeader title="Recent Alert Memory" sub="Select a backend alert to populate the feedback form" />
            {feedbackResource.loading && !feedbackPayload ? (
              <LoadingState message="Loading recent alerts…" />
            ) : feedbackResource.error ? (
              <ErrorState error={feedbackResource.error.message} onRetry={() => void feedbackResource.refetch()} />
            ) : recentAlerts.length === 0 ? (
              <EmptyState message="The backend returned no alert-memory rows." />
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full data-table">
                  <thead>
                    <tr>
                      <th>Unit</th><th>Cycle</th><th>Context</th><th>Alert</th>
                      <th>Anomaly Score</th><th>Root Cause</th><th>Feedback Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentAlerts.map((alert, index) => (
                      <tr
                        key={`${displayValue(alert.unit_id)}-${displayValue(alert.cycle)}-${index}`}
                        onClick={() => selectAlert(alert)}
                        className="cursor-pointer"
                        title="Use this alert in the feedback form"
                      >
                        <td className="font-mono font-semibold text-slate-200">{displayValue(alert.unit_id)}</td>
                        <td className="font-mono">{displayValue(alert.cycle)}</td>
                        <td className="font-mono text-slate-500">{displayValue(alert.context_id)}</td>
                        <td>{alert.alert_level ? <Badge label={String(alert.alert_level)} type="alert" /> : 'N/A'}</td>
                        <td>
                          <span
                            className="font-mono font-semibold"
                            style={{ color: getAlertColor(alert.alert_level) }}
                          >
                            {formatMetric(toFiniteNumber(alert.final_anomaly_score))}
                          </span>
                        </td>
                        <td className="text-xs text-slate-400">{displayValue(alert.root_cause_pattern)}</td>
                        <td className="text-xs font-mono text-slate-500">{humanize(feedbackLabel(alert))}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="glass p-5">
            <div className="flex items-center gap-2 mb-4">
              <Info size={14} className="text-cyan-400 flex-shrink-0" />
              <div>
                <div className="font-display font-bold text-xl text-slate-100">Adaptive Threshold Status</div>
                <div className="text-sm text-slate-500">Values loaded from the persisted threshold artifact</div>
              </div>
            </div>

            {thresholdResource.loading && !thresholdPayload ? (
              <LoadingState message="Loading adaptive thresholds…" />
            ) : thresholdResource.error ? (
              <ErrorState error={thresholdResource.error.message} onRetry={() => void thresholdResource.refetch()} />
            ) : thresholdGroups.length === 0 ? (
              <EmptyState message="No adaptive threshold artifact is available." />
            ) : (
              <>
                <div className="flex flex-wrap gap-x-5 gap-y-1 mb-5 text-[10px] font-mono text-slate-600">
                  <span>Artifact: {displayValue(thresholdPayload?.metadata?.name)}</span>
                  <span>Updated: {formatTimestamp(thresholdPayload?.metadata?.updated_at)}</span>
                  <span>Feedback used: {displayValue(thresholdContent?.feedback_records_used)}</span>
                  <span>Adjustment step: {displayValue(thresholdContent?.adjustment_step)}</span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3 mb-6">
                  {thresholdGroups.map(([context, levels]) => (
                    <div key={context} className="glass p-4 border-t-2 border-t-cyan-500/50">
                      <div className="text-[10px] font-mono text-slate-600 uppercase tracking-widest mb-2">
                        Context {context}
                      </div>
                      {Object.entries(levels).map(([level, value]) => (
                        <div key={level} className="flex justify-between gap-4 py-1.5 border-b border-[rgba(30,60,100,0.2)] last:border-0">
                          <span className="text-xs text-slate-500">{humanize(level)}</span>
                          <span className="font-mono text-xs text-cyan-400">
                            {formatMetric(toFiniteNumber(value), 6)}
                          </span>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>

                {thresholdContent?.rules && Object.keys(thresholdContent.rules).length > 0 && (
                  <div>
                    <div className="text-xs font-mono text-slate-600 uppercase tracking-widest mb-3">
                      Persisted Adaptation Rules
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {Object.entries(thresholdContent.rules).map(([label, rule]) => (
                        <div key={label} className="p-3 rounded-lg bg-[rgba(30,60,100,0.2)] border border-[rgba(30,60,100,0.3)]">
                          <div className="text-xs font-mono text-cyan-400">{humanize(label)}</div>
                          <div className="text-[11px] text-slate-500 mt-1">{displayValue(rule)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      <Disclaimer text="Operator feedback adapts detection thresholds; it does not schedule maintenance or confirm physical causality." />
    </div>
  )
}
