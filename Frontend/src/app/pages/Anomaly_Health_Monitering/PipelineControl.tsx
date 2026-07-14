import { useMemo, useState, type ReactNode } from 'react'
import {
  Activity,
  AlertTriangle,
  BarChart2,
  CheckCircle2,
  Clock,
  Cpu,
  Database,
  GitBranch,
  Lightbulb,
  Loader2,
  MessageSquare,
  Play,
  RefreshCw,
  Shield,
  XCircle,
} from 'lucide-react'
import { runFullPipeline, runPipelineStage } from '../../../api/Anomaly_Health_Monitering'
import { useApiMutation, usePipelineStatus } from '../../../hooks/Anomaly_Health_Monitering'
import type {
  ApiResponse,
  PipelineRunData,
  PipelineStageName,
} from '../../../types/Anomaly_Health_Monitering'
import { humanize } from '../../../utils/Anomaly_Health_Monitering/presentation'
import {
  Btn,
  Disclaimer,
  EmptyState,
  ErrorState,
  LoadingState,
  PageHeader,
  SectionHeader,
} from '../../components/ui/Anomaly_Health_Monitering/ui'

interface ArtifactSource {
  name?: string
  updated_at?: string | null
  size_bytes?: number | null
}

interface LivePipelineStage {
  sequence?: number
  id?: string
  name?: string
  status?: string
  status_source?: string
  primary_output?: string | null
  primary_output_exists?: boolean
  last_updated_at?: string | null
  records_count?: number | null
  duration_seconds?: number | null
  message?: string | null
  sources?: ArtifactSource[]
}

interface PipelineCounts {
  total?: number
  success?: number
  failed?: number
  not_run?: number
}

interface PipelineStatusPayload {
  overall_status?: string
  counts?: PipelineCounts
  success_count?: number
  failed_count?: number
  not_run_count?: number
  stages?: LivePipelineStage[]
}

type RunRequest =
  | { kind: 'full'; id: 'full' }
  | { kind: 'stage'; id: string; stage: PipelineStageName }

const STAGE_ACTIONS: Partial<Record<string, PipelineStageName>> = {
  preprocessing: 'preprocessing',
  context_modeling: 'contextModeling',
  digital_twin: 'digitalTwin',
  residual_analysis: 'residualAnalysis',
  anomaly_detection: 'anomalyDetection',
  health_monitoring: 'healthIndex',
  reasoning: 'reasoning',
  explainability: 'explainability',
  uncertainty: 'uncertainty',
  dashboard: 'dashboard',
}

const STAGE_ICONS: Record<string, ReactNode> = {
  preprocessing: <Database size={16} />,
  context_modeling: <GitBranch size={16} />,
  digital_twin: <Cpu size={16} />,
  residual_analysis: <Activity size={16} />,
  anomaly_detection: <AlertTriangle size={16} />,
  health_monitoring: <Activity size={16} />,
  reasoning: <GitBranch size={16} />,
  explainability: <Lightbulb size={16} />,
  uncertainty: <Shield size={16} />,
  feedback_learning: <MessageSquare size={16} />,
  dashboard: <BarChart2 size={16} />,
}

const GROUPS = [
  { id: 'model', label: 'Data and Model Layer' },
  { id: 'analysis', label: 'Analysis Layer' },
  { id: 'operations', label: 'Operations Layer' },
] as const

function groupForStage(stageId: string): (typeof GROUPS)[number]['id'] {
  if (['preprocessing', 'context_modeling', 'digital_twin'].includes(stageId)) return 'model'
  if (['residual_analysis', 'anomaly_detection', 'health_monitoring', 'reasoning'].includes(stageId)) {
    return 'analysis'
  }
  return 'operations'
}

function statusStyle(status: string) {
  const normalized = status.toLowerCase()
  if (normalized === 'success') return { color: '#22c55e', icon: <CheckCircle2 size={11} /> }
  if (normalized === 'failed') return { color: '#ef4444', icon: <XCircle size={11} /> }
  if (normalized === 'running') return { color: '#06b6d4', icon: <Loader2 size={11} className="animate-spin" /> }
  if (normalized === 'partial') return { color: '#f59e0b', icon: <AlertTriangle size={11} /> }
  return { color: '#64748b', icon: <Clock size={11} /> }
}

function StatusBadge({ status }: { status: string }) {
  const style = statusStyle(status)
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold"
      style={{ background: `${style.color}18`, color: style.color, border: `1px solid ${style.color}33` }}
    >
      {style.icon}{humanize(status)}
    </span>
  )
}

function formatTimestamp(value: unknown): string {
  if (!value) return 'N/A'
  const date = new Date(String(value))
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString()
}

function StageCard({
  stage,
  isRunning,
  pipelineBusy,
  onRun,
}: {
  stage: LivePipelineStage
  isRunning: boolean
  pipelineBusy: boolean
  onRun: (stageId: string, action: PipelineStageName) => void
}) {
  const stageId = String(stage.id ?? '')
  const persistedStatus = String(stage.status ?? 'not_run')
  const shownStatus = isRunning ? 'running' : persistedStatus
  const style = statusStyle(shownStatus)
  const action = STAGE_ACTIONS[stageId]

  return (
    <div className="glass p-4 flex flex-col gap-3 hover:border-[rgba(6,182,212,0.25)] transition-colors">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2.5 min-w-0">
          <div
            className="flex items-center justify-center w-8 h-8 rounded-lg flex-shrink-0"
            style={{ background: `${style.color}18`, color: style.color }}
          >
            {STAGE_ICONS[stageId] ?? <Activity size={16} />}
          </div>
          <div className="min-w-0">
            <div className="text-sm font-semibold text-slate-200 truncate">
              {stage.name ?? humanize(stageId)}
            </div>
            <div className="text-[10px] font-mono text-slate-600 mt-0.5 truncate">
              Output: {stage.primary_output ?? 'N/A'}
            </div>
          </div>
        </div>
        <StatusBadge status={shownStatus} />
      </div>

      <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px] font-mono">
        <span className="text-slate-700">Records</span>
        <span className="text-slate-500 text-right">
          {stage.records_count == null ? 'N/A' : stage.records_count.toLocaleString()}
        </span>
        <span className="text-slate-700">Last updated</span>
        <span className="text-slate-500 text-right truncate" title={String(stage.last_updated_at ?? '')}>
          {formatTimestamp(stage.last_updated_at)}
        </span>
        <span className="text-slate-700">Status source</span>
        <span className="text-slate-500 text-right truncate" title={stage.status_source}>
          {stage.status_source ?? 'N/A'}
        </span>
      </div>

      {stage.message && <div className="text-[10px] text-slate-600 leading-relaxed">{stage.message}</div>}

      <div className="flex items-center justify-between gap-3 mt-auto">
        <span className="text-[10px] font-mono text-slate-700">
          {(stage.sources?.length ?? 0).toLocaleString()} sources
        </span>
        {action ? (
          <button
            type="button"
            onClick={() => onRun(stageId, action)}
            disabled={pipelineBusy}
            className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            style={{ background: 'rgba(6,182,212,0.15)', color: '#06b6d4', border: '1px solid rgba(6,182,212,0.3)' }}
          >
            {isRunning ? <Loader2 size={11} className="animate-spin" /> : <Play size={11} />}
            {isRunning ? 'Running' : stageId === 'health_monitoring' ? 'Run Health Index' : 'Run'}
          </button>
        ) : (
          <span className="text-[10px] text-slate-700">No standalone endpoint</span>
        )}
      </div>
    </div>
  )
}

export default function PipelineControl() {
  const [activeRunId, setActiveRunId] = useState<string | null>(null)
  const pipelineResource = usePipelineStatus({ keepPreviousData: true })
  const payload = pipelineResource.data as unknown as PipelineStatusPayload | null
  const stages = useMemo(
    () => [...(payload?.stages ?? [])].sort((left, right) => (left.sequence ?? 0) - (right.sequence ?? 0)),
    [payload?.stages],
  )

  const counts: Required<PipelineCounts> = {
    total: payload?.counts?.total ?? stages.length,
    success: payload?.counts?.success ?? payload?.success_count ?? 0,
    failed: payload?.counts?.failed ?? payload?.failed_count ?? 0,
    not_run: payload?.counts?.not_run ?? payload?.not_run_count ?? 0,
  }

  const runMutation = useApiMutation<ApiResponse<PipelineRunData>, RunRequest>(
    async (request, signal) => {
      if (request.kind === 'full') return runFullPipeline({}, { signal })
      return runPipelineStage(request.stage, { signal })
    },
  )

  const execute = async (request: RunRequest) => {
    setActiveRunId(request.id)
    runMutation.reset()
    try {
      await runMutation.mutate(request)
      await pipelineResource.refetch()
    } catch {
      // The mutation hook exposes the backend error below the controls.
    } finally {
      setActiveRunId(null)
    }
  }

  const handleStageRun = (stageId: string, stage: PipelineStageName) => {
    void execute({ kind: 'stage', id: stageId, stage })
  }

  const busy = runMutation.loading

  return (
    <div className="page-content">
      <PageHeader
        title="Pipeline Control"
        subtitle="Inspect persisted backend stage state and invoke available processing endpoints."
        breadcrumb="System / Pipeline Control"
        actions={
          <div className="flex gap-2">
            <Btn
              variant="secondary"
              size="sm"
              disabled={pipelineResource.loading || busy}
              onClick={() => void pipelineResource.refetch()}
            >
              <RefreshCw size={13} /> Refresh
            </Btn>
            <Btn
              size="sm"
              disabled={busy}
              onClick={() => void execute({ kind: 'full', id: 'full' })}
            >
              {activeRunId === 'full' ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}
              {activeRunId === 'full' ? 'Running' : 'Full Pipeline'}
            </Btn>
          </div>
        }
      />

      <div className="flex items-start gap-3 p-4 rounded-xl mb-6 bg-amber-500/5 border border-amber-500/20">
        <AlertTriangle size={15} className="text-amber-400 mt-0.5 flex-shrink-0" />
        <div className="text-xs text-amber-400/80 leading-relaxed">
          Pipeline execution can be long-running. Stage state below is derived from generated backend
          artifacts and reports, then refreshed after each completed request.
        </div>
      </div>

      {runMutation.data && (
        <div className="flex items-start gap-2 p-3 rounded-lg mb-4 bg-green-500/10 border border-green-500/25 text-green-400 text-sm">
          <CheckCircle2 size={15} className="mt-0.5 flex-shrink-0" />
          <span>{runMutation.data.message}</span>
        </div>
      )}
      {runMutation.error && (
        <div className="p-3 rounded-lg mb-4 bg-red-500/10 border border-red-500/25 text-red-400 text-sm">
          {runMutation.error.message}
        </div>
      )}

      {pipelineResource.loading && !payload ? (
        <div className="glass p-5"><LoadingState message="Loading pipeline artifact status…" /></div>
      ) : pipelineResource.error ? (
        <ErrorState error={pipelineResource.error.message} onRetry={() => void pipelineResource.refetch()} />
      ) : stages.length === 0 ? (
        <div className="glass p-5"><EmptyState message="The backend returned no pipeline stages." /></div>
      ) : (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            {[
              { label: 'Overall Status', value: humanize(payload?.overall_status ?? 'not_run'), color: statusStyle(payload?.overall_status ?? 'not_run').color },
              { label: 'Successful', value: counts.success.toLocaleString(), color: '#22c55e' },
              { label: 'Failed', value: counts.failed.toLocaleString(), color: '#ef4444' },
              { label: 'Not Run', value: counts.not_run.toLocaleString(), color: '#64748b' },
            ].map((item) => (
              <div key={item.label} className="glass p-4" style={{ borderTop: `2px solid ${item.color}55` }}>
                <div className="text-[10px] font-mono text-slate-600 uppercase tracking-widest mb-2">{item.label}</div>
                <div className="font-display font-bold text-xl" style={{ color: item.color }}>{item.value}</div>
              </div>
            ))}
          </div>

          {GROUPS.map((group) => {
            const groupStages = stages.filter((stage) => groupForStage(String(stage.id ?? '')) === group.id)
            if (groupStages.length === 0) return null
            return (
              <div key={group.id} className="mb-6">
                <div className="text-[10px] font-mono text-slate-700 uppercase tracking-widest mb-3 flex items-center gap-2">
                  <div className="flex-1 h-px bg-[rgba(30,60,100,0.3)]" />
                  {group.label}
                  <div className="flex-1 h-px bg-[rgba(30,60,100,0.3)]" />
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
                  {groupStages.map((stage, index) => (
                    <StageCard
                      key={`${stage.id ?? 'stage'}-${stage.sequence ?? index}`}
                      stage={stage}
                      isRunning={activeRunId === stage.id}
                      pipelineBusy={busy}
                      onRun={handleStageRun}
                    />
                  ))}
                </div>
              </div>
            )
          })}

          <div className="glass p-5">
            <SectionHeader title="Artifact Activity" sub="Timestamps and messages reported by the backend" />
            <div className="space-y-2">
              {stages.map((stage, index) => (
                <div
                  key={`${stage.id ?? 'activity'}-${index}`}
                  className="grid grid-cols-[minmax(0,1fr)_auto] gap-4 p-3 rounded-lg bg-[rgba(4,10,22,0.45)] border border-[rgba(30,60,100,0.25)]"
                >
                  <div className="min-w-0">
                    <div className="text-xs font-mono text-slate-300 truncate">{stage.name ?? humanize(stage.id)}</div>
                    <div className="text-[10px] text-slate-600 mt-1 truncate">
                      {stage.message ?? stage.status_source ?? 'No report message available'}
                    </div>
                  </div>
                  <div className="text-[10px] font-mono text-slate-600 text-right">
                    {formatTimestamp(stage.last_updated_at)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      <Disclaimer text="Pipeline outputs support anomaly and health monitoring; they do not make maintenance scheduling decisions." />
    </div>
  )
}
