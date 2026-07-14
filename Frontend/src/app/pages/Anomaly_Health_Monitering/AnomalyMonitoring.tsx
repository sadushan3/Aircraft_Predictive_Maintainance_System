import { Fragment, useMemo, useState, type FormEvent } from 'react'
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { AlertTriangle, ChevronDown, ChevronUp, RefreshCw, Search } from 'lucide-react'
import {
  useDashboardAnalytics,
  useUnitAnomalies,
} from '../../../hooks/Anomaly_Health_Monitering'
import type { DashboardRecord } from '../../../types/Anomaly_Health_Monitering'
import {
  Badge,
  Disclaimer,
  EmptyState,
  FilterChip,
  PageHeader,
  SectionHeader,
  chartTooltipStyle,
} from '../../components/ui/Anomaly_Health_Monitering/ui'

const ALERT_FILTERS = ['All', 'Watch', 'Warning', 'Critical'] as const
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

function isNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function formatNumber(value: number | null | undefined, digits = 1): string {
  return isNumber(value) ? value.toFixed(digits) : '—'
}

function formatPercent(value: number | null | undefined, digits = 1): string {
  return isNumber(value) ? `${(value * 100).toFixed(digits)}%` : '—'
}

function formatCount(value: number | undefined): string {
  if (value === undefined) return '—'
  return new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value)
}

function unitLabel(unitId: number | null | undefined): string {
  return isNumber(unitId) ? `U-${String(unitId).padStart(3, '0')}` : '—'
}

function alertColor(level: string | null | undefined): string {
  return level ? ALERT_COLORS[level] ?? '#64748b' : '#64748b'
}

export default function AnomalyMonitoring() {
  const [unitInput, setUnitInput] = useState('')
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [inputError, setInputError] = useState<string | null>(null)
  const [filter, setFilter] = useState<(typeof ALERT_FILTERS)[number]>('All')
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const [expandedRow, setExpandedRow] = useState<string | null>(null)

  const analyticsResource = useDashboardAnalytics()
  const anomaliesResource = useUnitAnomalies(selectedId)
  const anomalies = anomaliesResource.data ?? []

  const analytics = asRecord(analyticsResource.data)
  const summaries = asRecord(analytics?.summaries)
  const dashboardSummary = asRecord(summaries?.dashboard_data_summary)
  const fleetAlertCounts = asRecord(dashboardSummary?.alert_counts)

  const filtered = useMemo(() => {
    const term = search.trim().toLowerCase()
    return anomalies.filter((row) => {
      if (filter !== 'All' && row.alert_level !== filter) return false
      if (!term) return true
      return [
        row.unit_id,
        row.cycle,
        row.alert_level,
        row.anomaly_status,
        row.root_cause_pattern,
        row.top_sensor_1,
        row.top_sensor_2,
        row.top_sensor_3,
      ].some((value) => String(value ?? '').toLowerCase().includes(term))
    })
  }, [anomalies, filter, search])

  const pageSize = 12
  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize))
  const currentPage = Math.min(page, totalPages)
  const pageRows = filtered.slice((currentPage - 1) * pageSize, currentPage * pageSize)

  const chartRows = useMemo(() => {
    const withCoordinates = filtered.filter((row) => isNumber(row.cycle) && isNumber(row.final_anomaly_score))
    const stride = Math.max(1, Math.ceil(withCoordinates.length / 400))
    return withCoordinates
      .filter((_, index) => index % stride === 0 || index === withCoordinates.length - 1)
      .map((row) => ({ cycle: row.cycle, anomaly_score: row.final_anomaly_score, alert_level: row.alert_level }))
  }, [filtered])

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
    setFilter('All')
    setSearch('')
    setPage(1)
    setExpandedRow(null)
  }

  const selectedCounts = useMemo(() => ({
    Watch: anomalies.filter((row) => row.alert_level === 'Watch').length,
    Warning: anomalies.filter((row) => row.alert_level === 'Warning').length,
    Critical: anomalies.filter((row) => row.alert_level === 'Critical').length,
  }), [anomalies])

  const refresh = () => {
    void analyticsResource.refetch()
    if (selectedId !== null) void anomaliesResource.refetch()
  }

  return (
    <div className="page-content">
      <PageHeader
        title="Anomaly Monitoring"
        subtitle="Fleet alert totals from report artifacts and anomaly records queried for a specific unit."
        breadcrumb="Monitoring / Anomaly Detection"
        actions={
          <button type="button" onClick={refresh} disabled={analyticsResource.loading || anomaliesResource.loading} className="inline-flex items-center gap-2 rounded-lg border border-[rgba(30,60,100,0.6)] bg-[rgba(30,60,100,0.4)] px-3 py-2 text-xs font-semibold text-slate-300 disabled:opacity-50">
            <RefreshCw size={13} className={analyticsResource.loading || anomaliesResource.loading ? 'animate-spin' : ''} /> Refresh
          </button>
        }
      />

      {(analyticsResource.error || anomaliesResource.error) && (
        <div className="mb-5 flex items-start gap-2 rounded-lg border border-red-500/20 bg-red-500/5 px-4 py-3 text-xs text-red-300">
          <AlertTriangle size={14} className="mt-0.5 shrink-0" />
          <span>{analyticsResource.error?.message ?? anomaliesResource.error?.message}</span>
        </div>
      )}

      <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
        {(['Watch', 'Warning', 'Critical'] as const).map((level) => {
          const fleetCount = numberAt(fleetAlertCounts, level)
          const color = alertColor(level)
          return (
            <div key={level} className="glass p-5" style={{ borderLeft: `3px solid ${color}` }}>
              <div className="text-xs font-mono uppercase tracking-widest" style={{ color }}>{level}</div>
              <div className="mt-2 font-display text-2xl font-bold text-slate-100">{formatCount(fleetCount)}</div>
              <div className="mt-1 text-xs text-slate-600">Fleet records in dashboard summary</div>
              {selectedId !== null && anomaliesResource.data !== null && (
                <div className="mt-3 border-t border-[rgba(30,60,100,0.3)] pt-3 text-xs text-slate-500">
                  {unitLabel(selectedId)}: <span className="font-mono text-slate-300">{selectedCounts[level]}</span>
                </div>
              )}
            </div>
          )
        })}
      </div>

      <form onSubmit={submitUnit} className="glass mb-6 p-4">
        <label htmlFor="anomaly-unit-id" className="mb-2 block text-[10px] font-mono uppercase tracking-widest text-slate-600">Unit ID</label>
        <div className="flex items-center gap-3">
          <Search size={14} className="text-slate-600" />
          <input
            id="anomaly-unit-id"
            value={unitInput}
            onChange={(event) => setUnitInput(event.target.value)}
            inputMode="numeric"
            placeholder="Enter a unit ID"
            className="w-full bg-transparent text-sm text-slate-300 outline-none placeholder:text-slate-600"
          />
          <button type="submit" className="shrink-0 rounded-lg bg-cyan-500 px-4 py-2 text-xs font-semibold text-[#040a16] hover:bg-cyan-400">Load Anomalies</button>
        </div>
        {inputError && <p className="mt-2 text-xs text-red-400">{inputError}</p>}
        <p className="mt-2 text-[11px] text-slate-600">The large dashboard dataset is queried only after you submit a unit ID.</p>
      </form>

      {selectedId === null ? (
        <EmptyState message="Enter a unit ID to request its anomaly records." />
      ) : anomaliesResource.loading && anomaliesResource.data === null ? (
        <div className="glass p-12 text-center text-sm text-slate-500">
          <RefreshCw className="mx-auto mb-3 animate-spin text-cyan-400" size={22} />
          Querying anomaly records for {unitLabel(selectedId)}…
        </div>
      ) : anomalies.length === 0 ? (
        <div className="glass p-5">
          <EmptyState message={`No Watch, Warning, or Critical records were returned for ${unitLabel(selectedId)}.`} />
        </div>
      ) : (
        <>
          <div className="glass mb-6 p-5">
            <SectionHeader title="Anomaly Score Timeline" sub={`${unitLabel(selectedId)} · ${filtered.length} matching backend records`} />
            {chartRows.length === 0 ? <EmptyState message="No numeric anomaly scores were returned for this selection." /> : (
              <ResponsiveContainer width="100%" height={270}>
                <AreaChart data={chartRows} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="anomalyScoreFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ef4444" stopOpacity={0.35} />
                      <stop offset="100%" stopColor="#ef4444" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="rgba(30,60,100,0.25)" vertical={false} />
                  <XAxis dataKey="cycle" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={chartTooltipStyle} />
                  <Area type="monotone" dataKey="anomaly_score" name="Anomaly score" stroke="#ef4444" strokeWidth={2} fill="url(#anomalyScoreFill)" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>

          <div className="glass mb-4 flex flex-wrap items-center gap-3 p-4">
            <div className="flex flex-wrap gap-2">
              {ALERT_FILTERS.map((level) => (
                <FilterChip key={level} label={level} active={filter === level} color={level === 'All' ? undefined : alertColor(level)} onClick={() => { setFilter(level); setPage(1) }} />
              ))}
            </div>
            <input
              value={search}
              onChange={(event) => { setSearch(event.target.value); setPage(1) }}
              placeholder="Filter returned records…"
              className="ml-auto w-56 rounded-lg border border-[rgba(30,60,100,0.4)] bg-[rgba(30,60,100,0.3)] px-3 py-2 text-xs font-mono text-slate-300 outline-none placeholder:text-slate-600"
            />
          </div>

          <div className="glass overflow-hidden">
            {filtered.length === 0 ? <EmptyState message="No returned anomaly records match these filters." /> : (
              <div className="overflow-x-auto">
                <table className="data-table w-full">
                  <thead>
                    <tr><th>Unit</th><th>Cycle</th><th>Split</th><th>Alert</th><th>Anomaly Score</th><th>Health</th><th>Confidence</th><th>Pattern</th><th>Top Sensor</th><th>Detail</th></tr>
                  </thead>
                  <tbody>
                    {pageRows.map((row: DashboardRecord, index) => {
                      const rowKey = `${row.unit_id ?? selectedId}-${row.cycle ?? 'unknown'}-${(currentPage - 1) * pageSize + index}`
                      return (
                        <Fragment key={rowKey}>
                          <tr className="cursor-pointer" onClick={() => setExpandedRow(expandedRow === rowKey ? null : rowKey)}>
                            <td><span className="font-mono font-semibold text-slate-200">{unitLabel(row.unit_id)}</span></td>
                            <td><span className="font-mono">{formatNumber(row.cycle, 0)}</span></td>
                            <td><span className="font-mono text-slate-500">{row.split ?? '—'}</span></td>
                            <td>{row.alert_level ? <Badge label={row.alert_level} type="alert" /> : '—'}</td>
                            <td><span className="font-mono font-semibold" style={{ color: alertColor(row.alert_level) }}>{formatNumber(row.final_anomaly_score, 4)}</span></td>
                            <td>{row.health_state ? <Badge label={row.health_state} type="health" /> : '—'}</td>
                            <td><span className="font-mono text-slate-400">{formatPercent(row.confidence_score)}</span></td>
                            <td><span className="block max-w-40 truncate text-xs text-slate-500">{row.root_cause_pattern?.replace(/_/g, ' ') ?? '—'}</span></td>
                            <td><span className="font-mono text-cyan-400">{row.top_sensor_1 ?? '—'}</span></td>
                            <td>{expandedRow === rowKey ? <ChevronUp size={14} /> : <ChevronDown size={14} />}</td>
                          </tr>
                          {expandedRow === rowKey && (
                            <tr>
                              <td colSpan={10} className="bg-[rgba(6,182,212,0.03)]">
                                <div className="grid grid-cols-1 gap-5 p-4 lg:grid-cols-3">
                                  <div>
                                    <div className="mb-2 text-[10px] font-mono uppercase tracking-widest text-slate-600">Sensor Contributions</div>
                                    {[
                                      [row.top_sensor_1, row.contribution_1],
                                      [row.top_sensor_2, row.contribution_2],
                                      [row.top_sensor_3, row.contribution_3],
                                    ].map(([sensor, contribution], sensorIndex) => (
                                      <div key={`${String(sensor)}-${sensorIndex}`} className="flex justify-between py-1 text-xs">
                                        <span className="font-mono text-slate-300">{typeof sensor === 'string' ? sensor : '—'}</span>
                                        <span className="font-mono text-slate-500">{typeof contribution === 'number' ? formatPercent(contribution) : '—'}</span>
                                      </div>
                                    ))}
                                  </div>
                                  <div>
                                    <div className="mb-2 text-[10px] font-mono uppercase tracking-widest text-slate-600">Inspection Focus</div>
                                    <p className="text-xs leading-relaxed text-slate-400">{row.inspection_focus ?? 'No inspection focus was returned.'}</p>
                                  </div>
                                  <div>
                                    <div className="mb-2 text-[10px] font-mono uppercase tracking-widest text-slate-600">Explanation</div>
                                    <p className="text-xs leading-relaxed text-slate-500">{row.explanation_text ?? 'No explanation was returned.'}</p>
                                  </div>
                                </div>
                              </td>
                            </tr>
                          )}
                        </Fragment>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
            {filtered.length > 0 && (
              <div className="flex items-center justify-between border-t border-[rgba(30,60,100,0.3)] px-4 py-3">
                <span className="text-xs font-mono text-slate-600">{filtered.length} records · Page {currentPage} of {totalPages}</span>
                <div className="flex gap-2">
                  <button type="button" disabled={currentPage === 1} onClick={() => setPage((value) => Math.max(1, value - 1))} className="rounded bg-[rgba(30,60,100,0.3)] px-3 py-1 text-xs font-mono text-slate-400 disabled:opacity-30">← Prev</button>
                  <button type="button" disabled={currentPage === totalPages} onClick={() => setPage((value) => Math.min(totalPages, value + 1))} className="rounded bg-[rgba(30,60,100,0.3)] px-3 py-1 text-xs font-mono text-slate-400 disabled:opacity-30">Next →</button>
                </div>
              </div>
            )}
          </div>
        </>
      )}

      <Disclaimer text="Anomaly records and explanations support inspection focus only; they do not make maintenance scheduling decisions." />
    </div>
  )
}
