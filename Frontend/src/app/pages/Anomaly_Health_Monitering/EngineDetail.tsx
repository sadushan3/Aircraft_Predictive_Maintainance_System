import { useMemo, useState, type FormEvent } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { AlertTriangle, RefreshCw, Search } from 'lucide-react'
import {
  useHealthTrend,
  useLatestUnit,
} from '../../../hooks/Anomaly_Health_Monitering'
import {
  Badge,
  CircularGauge,
  Disclaimer,
  EmptyState,
  MetricRow,
  PageHeader,
  SectionHeader,
  SensorBar,
  chartTooltipStyle,
} from '../../components/ui/Anomaly_Health_Monitering/ui'

function isNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function formatNumber(value: number | null | undefined, digits = 1): string {
  return isNumber(value) ? value.toFixed(digits) : '—'
}

function formatPercent(value: number | null | undefined, digits = 1): string {
  return isNumber(value) ? `${(value * 100).toFixed(digits)}%` : '—'
}

function healthIndexColor(value: number | null | undefined): string {
  if (!isNumber(value)) return '#64748b'
  if (value >= 85) return '#22c55e'
  if (value >= 65) return '#3b82f6'
  if (value >= 40) return '#f59e0b'
  return '#ef4444'
}

function unitLabel(unitId: number | null | undefined): string {
  return isNumber(unitId) ? `U-${String(unitId).padStart(3, '0')}` : '—'
}

export default function EngineDetail() {
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [unitInput, setUnitInput] = useState('')
  const [inputError, setInputError] = useState<string | null>(null)

  const unitResource = useLatestUnit(selectedId)
  const trendResource = useHealthTrend(selectedId)
  const unit = unitResource.data
  const trend = trendResource.data ?? []

  const trendData = useMemo(
    () => trend
      .filter((row) => isNumber(row.cycle))
      .map((row) => ({
        cycle: row.cycle,
        health_index: isNumber(row.health_index) ? row.health_index : null,
        anomaly_score: isNumber(row.final_anomaly_score) ? row.final_anomaly_score : null,
      })),
    [trend],
  )

  const sensors = useMemo(() => {
    if (!unit) return []
    const candidates = [
      { sensor: unit.top_sensor_1, contribution: unit.contribution_1, rank: 1, color: '#06b6d4' },
      { sensor: unit.top_sensor_2, contribution: unit.contribution_2, rank: 2, color: '#3b82f6' },
      { sensor: unit.top_sensor_3, contribution: unit.contribution_3, rank: 3, color: '#8b5cf6' },
    ]
    return candidates.filter(
      (item): item is { sensor: string; contribution: number; rank: number; color: string } =>
        typeof item.sensor === 'string' && item.sensor.length > 0 && isNumber(item.contribution),
    )
  }, [unit])

  const refresh = () => {
    if (selectedId !== null) {
      void unitResource.refetch()
      void trendResource.refetch()
    }
  }

  const selectedLoading = selectedId !== null && (unitResource.loading || trendResource.loading)
  const loadError = unitResource.error ?? trendResource.error

  const submitUnit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const normalized = unitInput.trim()
    const parsed = Number(normalized)
    if (!normalized || !Number.isInteger(parsed) || parsed < 0) {
      setInputError('Enter a valid non-negative unit ID.')
      return
    }
    setInputError(null)
    setSelectedId(parsed)
  }

  return (
    <div className="page-content">
      <PageHeader
        title="Engine Unit Detail"
        subtitle="Latest health, anomaly, confidence, explanation, and health history returned by the backend."
        breadcrumb="Monitoring / Engine Detail"
        actions={
          <button
            type="button"
            onClick={refresh}
            disabled={selectedId === null || unitResource.loading || trendResource.loading}
            className="inline-flex items-center gap-2 rounded-lg border border-[rgba(30,60,100,0.6)] bg-[rgba(30,60,100,0.4)] px-3 py-2 text-xs font-semibold text-slate-300 disabled:opacity-50"
          >
            <RefreshCw size={13} className={selectedLoading ? 'animate-spin' : ''} /> Refresh
          </button>
        }
      />

      {loadError && (
        <div className="mb-5 flex items-start gap-2 rounded-lg border border-red-500/20 bg-red-500/5 px-4 py-3 text-xs text-red-300">
          <AlertTriangle size={14} className="mt-0.5 shrink-0" />
          <span>{loadError.message}</span>
        </div>
      )}

      <form onSubmit={submitUnit} className="glass mb-6 p-4">
        <label htmlFor="engine-unit-id" className="mb-2 block text-[10px] font-mono uppercase tracking-widest text-slate-600">
          Unit ID
        </label>
        <div className="flex items-center gap-3">
          <Search size={14} className="text-slate-600" />
          <input
            id="engine-unit-id"
            value={unitInput}
            onChange={(event) => setUnitInput(event.target.value)}
            inputMode="numeric"
            placeholder="Enter a unit ID"
            className="w-full bg-transparent text-sm text-slate-300 outline-none placeholder:text-slate-600"
          />
          <button type="submit" className="shrink-0 rounded-lg bg-cyan-500 px-4 py-2 text-xs font-semibold text-[#040a16] hover:bg-cyan-400">
            Load Unit
          </button>
        </div>
        {inputError && <p className="mt-2 text-xs text-red-400">{inputError}</p>}
        <p className="mt-2 text-[11px] text-slate-600">The backend query begins only after you submit a unit ID.</p>
      </form>

      {selectedId === null ? (
        <EmptyState message="Enter a unit ID to request its latest record and health history." />
      ) : selectedLoading && !unit ? (
        <div className="glass p-12 text-center text-sm text-slate-500">
          <RefreshCw className="mx-auto mb-3 animate-spin text-cyan-400" size={22} />
          Loading {unitLabel(selectedId)} from the backend…
        </div>
      ) : !unit || !isNumber(unit.unit_id) ? (
        <EmptyState message={`No latest backend record was returned for ${unitLabel(selectedId)}.`} />
      ) : (
        <>
          <div className="mb-6 grid grid-cols-1 gap-4 lg:grid-cols-3">
            <div className="glass p-5">
              <SectionHeader title={unitLabel(unit.unit_id)} sub={`Latest cycle ${formatNumber(unit.cycle, 0)}`} />
              <div className="flex justify-center">
                {isNumber(unit.health_index) ? (
                  <CircularGauge value={unit.health_index} max={100} color={healthIndexColor(unit.health_index)} label="Health Index" sub="OUT OF 100" />
                ) : (
                  <EmptyState message="Health index unavailable." />
                )}
              </div>
              <div className="mt-2 flex justify-center gap-2">
                {unit.health_state && <Badge label={unit.health_state} type="health" />}
                {unit.alert_level && <Badge label={unit.alert_level} type="alert" />}
              </div>
            </div>

            <div className="glass p-5">
              <SectionHeader title="Latest Scores" sub="Latest unit record" />
              <MetricRow label="Anomaly score" value={formatNumber(unit.final_anomaly_score, 4)} />
              <MetricRow label="Remaining health" value={isNumber(unit.remaining_health_percentage) ? `${formatNumber(unit.remaining_health_percentage)}%` : '—'} color={healthIndexColor(unit.remaining_health_percentage)} />
              <MetricRow label="Confidence" value={formatPercent(unit.confidence_score)} color="#06b6d4" />
              <MetricRow label="Uncertainty" value={formatPercent(unit.uncertainty_score)} color="#a855f7" />
              <MetricRow label="Reliability" value={formatPercent(unit.reliability_score)} color="#22c55e" />
              <MetricRow label="Data split" value={unit.split ?? '—'} />
            </div>

            <div className="glass p-5">
              <SectionHeader title="Sensor Contributions" sub="Latest root-cause ranking" />
              {sensors.length === 0 ? (
                <EmptyState message="No sensor contribution ranking was returned." />
              ) : (
                sensors.map((sensor) => (
                  <SensorBar key={`${sensor.rank}-${sensor.sensor}`} {...sensor} />
                ))
              )}
            </div>
          </div>

          <div className="glass mb-6 p-5">
            <SectionHeader title="Health and Anomaly History" sub={`${unitLabel(unit.unit_id)} · backend health-trend endpoint`} />
            {trendResource.loading && trendResource.data === null ? (
              <div className="py-14 text-center text-sm text-slate-500">Loading health history…</div>
            ) : trendData.length === 0 ? (
              <EmptyState message="No health history was returned for this unit." />
            ) : (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={trendData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="rgba(30,60,100,0.25)" vertical={false} />
                  <XAxis dataKey="cycle" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis yAxisId="health" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis yAxisId="score" orientation="right" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={chartTooltipStyle} />
                  <Line yAxisId="health" type="monotone" dataKey="health_index" name="Health index" stroke="#06b6d4" strokeWidth={2} dot={false} />
                  <Line yAxisId="score" type="monotone" dataKey="anomaly_score" name="Anomaly score" stroke="#f97316" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <div className="glass p-5">
              <SectionHeader title="Reasoning Output" sub="Latest root-cause result" />
              <MetricRow label="Pattern" value={unit.root_cause_pattern?.replace(/_/g, ' ') ?? '—'} />
              <MetricRow label="Primary subsystem" value={unit.primary_subsystem?.replace(/_/g, ' ') ?? '—'} />
              {unit.inspection_focus ? (
                <p className="mt-4 text-sm leading-relaxed text-slate-400">{unit.inspection_focus}</p>
              ) : (
                <EmptyState message="No inspection focus was returned." />
              )}
            </div>
            <div className="glass p-5">
              <SectionHeader title="Explanation" sub="Generated explanation for the latest cycle" />
              {unit.explanation_text ? (
                <p className="text-sm leading-relaxed text-slate-400">{unit.explanation_text}</p>
              ) : unit.subsystem_explanation ? (
                <p className="text-sm leading-relaxed text-slate-400">{unit.subsystem_explanation}</p>
              ) : (
                <EmptyState message="No explanation was returned." />
              )}
            </div>
          </div>
        </>
      )}

      <Disclaimer text="This display reports backend model outputs for inspection support; it does not make maintenance scheduling decisions." />
    </div>
  )
}
