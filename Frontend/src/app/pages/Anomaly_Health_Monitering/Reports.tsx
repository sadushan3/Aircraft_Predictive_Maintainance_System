import { useMemo, useState } from 'react'
import { FileBarChart, FileText, RefreshCw } from 'lucide-react'
import { useReports } from '../../../hooks/Anomaly_Health_Monitering'
import { humanize } from '../../../utils/Anomaly_Health_Monitering/presentation'
import {
  Btn,
  EmptyState,
  ErrorState,
  LoadingState,
  PageHeader,
  SectionHeader,
} from '../../components/ui/Anomaly_Health_Monitering/ui'

interface CatalogFile {
  name?: string
  extension?: string
  size_bytes?: number | null
  updated_at?: string | null
  records_count?: number | null
}

interface CatalogReport extends CatalogFile {
  status?: string
  message?: string | null
  content?: unknown
  content_status?: string
}

interface ReportCatalogPayload {
  reports?: CatalogReport[]
  output_files?: CatalogFile[]
  metric_files?: CatalogFile[]
  report_count?: number
  output_file_count?: number
  metric_file_count?: number
  truncated?: boolean
}

interface FlatMetric {
  path: string
  value: string
}

function formatBytes(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return 'N/A'
  if (value < 1024) return `${value.toLocaleString()} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let size = value / 1024
  let unitIndex = 0
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex += 1
  }
  return `${size.toFixed(size >= 10 ? 1 : 2)} ${units[unitIndex]}`
}

function formatTimestamp(value: unknown): string {
  if (!value) return 'N/A'
  const date = new Date(String(value))
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString()
}

function scalarValue(value: unknown): string {
  if (value === null || value === undefined) return 'N/A'
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return 'N/A'
    return value.toLocaleString(undefined, { maximumFractionDigits: 8 })
  }
  if (typeof value === 'string') return value || 'N/A'
  return String(value)
}

function flattenContent(value: unknown, path = '', depth = 0): FlatMetric[] {
  const label = path || 'value'
  if (value === null || value === undefined || typeof value !== 'object') {
    return [{ path: label, value: scalarValue(value) }]
  }
  if (depth >= 5) {
    return [{ path: label, value: Array.isArray(value) ? `${value.length} entries` : 'Nested object' }]
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return [{ path: label, value: 'Empty' }]
    return value.flatMap((item, index) => flattenContent(item, `${label}[${index}]`, depth + 1))
  }
  const entries = Object.entries(value as Record<string, unknown>)
  if (entries.length === 0) return [{ path: label, value: 'Empty' }]
  return entries.flatMap(([key, nested]) => (
    flattenContent(nested, path ? `${path}.${key}` : key, depth + 1)
  ))
}

function fileKey(file: CatalogFile, index: number): string {
  return `${file.name ?? 'artifact'}-${file.updated_at ?? index}`
}

export default function Reports() {
  const [selectedReportName, setSelectedReportName] = useState('')
  const reportsResource = useReports({ keepPreviousData: true })
  const payload = reportsResource.data as unknown as ReportCatalogPayload | null
  const reports = payload?.reports ?? []
  const outputFiles = payload?.output_files ?? []
  const metricFiles = payload?.metric_files ?? []
  const selectedReport = reports.find((report) => report.name === selectedReportName) ?? reports[0]
  const flattenedContent = useMemo(
    () => flattenContent(selectedReport?.content).slice(0, 150),
    [selectedReport],
  )
  const totalFlattenedMetrics = useMemo(
    () => flattenContent(selectedReport?.content).length,
    [selectedReport],
  )
  const noCatalogData = reports.length === 0 && outputFiles.length === 0 && metricFiles.length === 0

  const countCards = [
    { label: 'JSON Reports', value: payload?.report_count ?? reports.length, color: '#a855f7' },
    { label: 'Output Artifacts', value: payload?.output_file_count ?? outputFiles.length, color: '#06b6d4' },
    { label: 'Metric Artifacts', value: payload?.metric_file_count ?? metricFiles.length, color: '#22c55e' },
    {
      label: 'Catalog Total',
      value: (payload?.report_count ?? reports.length)
        + (payload?.output_file_count ?? outputFiles.length)
        + (payload?.metric_file_count ?? metricFiles.length),
      color: '#f59e0b',
    },
  ]

  return (
    <div className="page-content">
      <PageHeader
        title="Reports & Metrics"
        subtitle="Live report content and artifact metadata returned by the backend catalog."
        breadcrumb="System / Reports & Metrics"
        actions={
          <Btn
            size="sm"
            variant="secondary"
            disabled={reportsResource.loading}
            onClick={() => void reportsResource.refetch()}
          >
            <RefreshCw size={13} /> Refresh Catalog
          </Btn>
        }
      />

      {reportsResource.loading && !payload ? (
        <div className="glass p-5"><LoadingState message="Loading report catalog…" /></div>
      ) : reportsResource.error ? (
        <ErrorState error={reportsResource.error.message} onRetry={() => void reportsResource.refetch()} />
      ) : noCatalogData ? (
        <div className="glass p-5"><EmptyState message="No generated reports or artifacts were found by the backend." /></div>
      ) : (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            {countCards.map(({ label, value, color }) => (
              <div key={label} className="glass p-4" style={{ borderTop: `2px solid ${color}55` }}>
                <div className="text-[10px] font-mono text-slate-600 uppercase tracking-widest mb-2">{label}</div>
                <div className="font-display font-bold text-2xl" style={{ color }}>{value.toLocaleString()}</div>
              </div>
            ))}
          </div>

          {payload?.truncated && (
            <div className="p-3 rounded-lg mb-4 bg-amber-500/5 border border-amber-500/20 text-amber-400/80 text-xs">
              The backend bounded this catalog response. Counts and files shown are the returned subset.
            </div>
          )}

          <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 mb-6">
            <div className="glass p-5">
              <SectionHeader title="Generated Reports" sub="Select a JSON report to inspect its persisted content" />
              {reports.length === 0 ? (
                <EmptyState message="No JSON report files are available." />
              ) : (
                <div className="space-y-2 max-h-[560px] overflow-y-auto pr-1">
                  {reports.map((report, index) => {
                    const active = report.name === selectedReport?.name
                    const status = report.status ?? 'available'
                    return (
                      <button
                        type="button"
                        key={fileKey(report, index)}
                        onClick={() => setSelectedReportName(report.name ?? '')}
                        className="w-full text-left p-3 rounded-lg transition-colors"
                        style={{
                          background: active ? 'rgba(6,182,212,0.1)' : 'rgba(30,60,100,0.12)',
                          border: `1px solid ${active ? 'rgba(6,182,212,0.35)' : 'rgba(30,60,100,0.25)'}`,
                        }}
                      >
                        <div className="flex items-center gap-2">
                          <FileBarChart size={14} className={active ? 'text-cyan-400' : 'text-slate-600'} />
                          <span className="font-mono text-xs text-slate-300 truncate flex-1">{report.name ?? 'Unnamed report'}</span>
                          <span className="text-[9px] font-mono text-slate-600">{humanize(status)}</span>
                        </div>
                        <div className="flex justify-between gap-3 mt-2 text-[10px] font-mono text-slate-700">
                          <span>{formatBytes(report.size_bytes)}</span>
                          <span className="truncate">{formatTimestamp(report.updated_at)}</span>
                        </div>
                      </button>
                    )
                  })}
                </div>
              )}
            </div>

            <div className="glass p-5 xl:col-span-2">
              <SectionHeader
                title={selectedReport?.name ?? 'Report Content'}
                sub="Scalar values are rendered exactly from the selected backend JSON report"
              />
              {!selectedReport ? (
                <EmptyState message="Select an available report." />
              ) : selectedReport.content == null ? (
                <div className="space-y-4">
                  <EmptyState message={`Report content is unavailable (${selectedReport.content_status ?? 'no content status'}).`} />
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <span className="text-slate-600">Status</span>
                    <span className="font-mono text-slate-300 text-right">{humanize(selectedReport.status)}</span>
                    <span className="text-slate-600">Records</span>
                    <span className="font-mono text-slate-300 text-right">
                      {selectedReport.records_count == null ? 'N/A' : selectedReport.records_count.toLocaleString()}
                    </span>
                  </div>
                </div>
              ) : (
                <>
                  {selectedReport.message && (
                    <div className="p-3 rounded-lg mb-4 bg-[rgba(30,60,100,0.2)] border border-[rgba(30,60,100,0.3)] text-xs text-slate-400">
                      {selectedReport.message}
                    </div>
                  )}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 max-h-[530px] overflow-y-auto pr-2">
                    {flattenedContent.map((metric, index) => (
                      <div
                        key={`${metric.path}-${index}`}
                        className="flex items-start justify-between gap-4 py-2.5 border-b border-[rgba(30,60,100,0.25)]"
                      >
                        <span className="text-xs text-slate-500 break-all">{metric.path}</span>
                        <span className="font-mono text-xs text-slate-200 text-right break-all max-w-[55%]">{metric.value}</span>
                      </div>
                    ))}
                  </div>
                  {totalFlattenedMetrics > flattenedContent.length && (
                    <div className="mt-3 text-[10px] font-mono text-slate-600">
                      Showing {flattenedContent.length.toLocaleString()} of {totalFlattenedMetrics.toLocaleString()} scalar values.
                    </div>
                  )}
                </>
              )}
            </div>
          </div>

          {[
            { title: 'Output Artifacts', files: outputFiles, color: '#06b6d4' },
            { title: 'Metric Artifacts', files: metricFiles, color: '#22c55e' },
          ].map(({ title, files, color }) => (
            <div key={title} className="glass p-5 mb-4">
              <SectionHeader title={title} sub="File metadata exposed by the backend; file downloads are not enabled" />
              {files.length === 0 ? (
                <EmptyState message={`No ${title.toLowerCase()} are available.`} />
              ) : (
                <div className="space-y-1">
                  {files.map((file, index) => (
                    <div
                      key={fileKey(file, index)}
                      className="grid grid-cols-[auto_minmax(0,1fr)_auto_auto_auto] items-center gap-4 py-2.5 px-3 rounded-lg hover:bg-[rgba(30,60,100,0.2)] transition-colors"
                    >
                      <FileText size={14} style={{ color }} />
                      <span className="font-mono text-sm text-slate-300 truncate">{file.name ?? 'Unnamed artifact'}</span>
                      <span className="font-mono text-xs text-slate-600">
                        {file.records_count == null ? 'N/A records' : `${file.records_count.toLocaleString()} records`}
                      </span>
                      <span className="font-mono text-xs text-slate-600">{formatBytes(file.size_bytes)}</span>
                      <span
                        className="px-1.5 py-0.5 rounded text-[10px] font-mono uppercase"
                        style={{ background: `${color}18`, color, border: `1px solid ${color}33` }}
                      >
                        {file.extension ?? 'file'}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </>
      )}
    </div>
  )
}
