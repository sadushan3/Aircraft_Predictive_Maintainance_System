import type { ReactNode } from 'react'
import { NavLink } from 'react-router-dom'
import {
  Activity,
  AlertTriangle,
  BarChart2,
  Cpu,
  Database,
  GitBranch,
  Layers,
  LayoutDashboard,
  Lightbulb,
  MessageSquare,
  Shield,
  Zap,
} from 'lucide-react'

import { useDashboardSummary, usePipelineStatus } from '../../../../hooks/Anomaly_Health_Monitering'
import { formatCount } from '../../../../utils/Anomaly_Health_Monitering/presentation'

const NAV_GROUPS = [
  {
    label: 'Monitoring',
    items: [
      { path: '/overview', label: 'Overview', icon: LayoutDashboard },
      { path: '/engine', label: 'Engine Detail', icon: Cpu },
      { path: '/anomaly', label: 'Anomaly Monitoring', icon: AlertTriangle },
      { path: '/health', label: 'Health Monitoring', icon: Activity },
      { path: '/rootcause', label: 'Root-Cause Reasoning', icon: GitBranch },
    ],
  },
  {
    label: 'Intelligence',
    items: [
      { path: '/confidence', label: 'Confidence & Uncertainty', icon: Shield },
      { path: '/explain', label: 'Explainability', icon: Lightbulb },
      { path: '/feedback', label: 'Feedback Learning', icon: MessageSquare },
    ],
  },
  {
    label: 'System',
    items: [
      { path: '/pipeline', label: 'Pipeline Control', icon: Layers },
      { path: '/reports', label: 'Reports & Metrics', icon: BarChart2 },
    ],
  },
]

function stageLabel(stage: Record<string, unknown>): string {
  return String(stage.name ?? stage.stage ?? 'Pipeline stage')
}

function stageStatus(stage: Record<string, unknown>): string {
  return String(stage.status ?? 'unknown')
}

export default function Layout({ children }: { children: ReactNode }) {
  const summary = useDashboardSummary()
  const pipeline = usePipelineStatus()
  const stages = Array.isArray(pipeline.data?.stages) ? pipeline.data.stages.slice(0, 3) : []
  const criticalAlerts = summary.data?.alert_counts?.critical ?? 0
  const pipelineState = pipeline.loading
    ? 'Loading'
    : pipeline.error
      ? 'Unavailable'
      : String(pipeline.data?.overall_status ?? pipeline.data?.status ?? 'Ready')

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: '#040a16' }}>
      <aside
        className="flex flex-col flex-shrink-0 h-full overflow-hidden"
        style={{
          width: 228,
          background: 'linear-gradient(180deg, #060c1a 0%, #050b17 100%)',
          borderRight: '1px solid rgba(30,60,100,0.4)',
        }}
      >
        <div className="flex items-center gap-2.5 px-5 py-4 border-b border-[rgba(30,60,100,0.35)]">
          <div
            className="flex items-center justify-center rounded-lg"
            style={{
              width: 32,
              height: 32,
              background: 'linear-gradient(135deg, #06b6d4, #3b82f6)',
              boxShadow: '0 0 14px rgba(6,182,212,0.4)',
            }}
          >
            <Zap size={16} color="white" />
          </div>
          <div>
            <div className="font-display font-bold text-xs text-slate-100 leading-tight">CA-EDT-AHMA</div>
            <div className="text-[9px] text-slate-600 font-mono leading-tight">Predictive Maintenance</div>
          </div>
        </div>

        <nav className="flex-1 overflow-y-auto py-3 px-3 space-y-3">
          {NAV_GROUPS.map((group) => (
            <div key={group.label}>
              <div className="px-2 pb-1 pt-1 text-[9px] font-mono text-slate-700 uppercase tracking-widest">
                {group.label}
              </div>
              {group.items.map(({ path, label, icon: Icon }) => (
                <NavLink
                  key={path}
                  to={path}
                  className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                >
                  <Icon size={15} />
                  <span>{label}</span>
                </NavLink>
              ))}
            </div>
          ))}
        </nav>

        <div className="border-t border-[rgba(30,60,100,0.35)] p-3">
          <div className="text-[9px] font-mono text-slate-700 uppercase tracking-widest mb-2 px-1">Backend stages</div>
          {pipeline.loading && <div className="px-1 py-1 text-[11px] text-slate-600">Loading status…</div>}
          {pipeline.error && <div className="px-1 py-1 text-[11px] text-red-400">Status unavailable</div>}
          {!pipeline.loading && !pipeline.error && stages.length === 0 && (
            <div className="px-1 py-1 text-[11px] text-slate-600">No stage reports found</div>
          )}
          {stages.map((stage, index) => {
            const status = stageStatus(stage)
            const ok = status.toLowerCase() === 'success'
            return (
              <div key={`${stageLabel(stage)}-${index}`} className="flex items-center justify-between gap-2 px-1 py-1">
                <span className="text-[11px] text-slate-600 truncate">{stageLabel(stage)}</span>
                <span className={`text-[10px] font-mono ${ok ? 'text-green-500' : 'text-amber-500'}`}>
                  {status}
                </span>
              </div>
            )
          })}
        </div>
      </aside>

      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <header
          className="flex items-center justify-between px-6 flex-shrink-0"
          style={{
            height: 54,
            background: 'rgba(6,12,26,0.9)',
            borderBottom: '1px solid rgba(30,60,100,0.4)',
            backdropFilter: 'blur(12px)',
          }}
        >
          <div className="flex items-center gap-5 text-xs font-mono">
            <div className="flex items-center gap-1.5">
              <Database size={12} className="text-slate-600" />
              <span className="text-slate-500">
                {summary.loading ? 'Loading records…' : `${formatCount(summary.data?.total_records)} records`}
              </span>
            </div>
            <div className="text-slate-600">
              Units: <span className="text-slate-400">{formatCount(summary.data?.unique_units)}</span>
            </div>
          </div>

          <div className="flex items-center gap-5 text-xs font-mono">
            <div className="text-slate-600">
              Pipeline: <span className={pipeline.error ? 'text-red-400' : 'text-cyan-400'}>{pipelineState}</span>
            </div>
            <div className="flex items-center gap-1.5 text-slate-600">
              <AlertTriangle size={13} className={criticalAlerts > 0 ? 'text-red-400' : 'text-slate-600'} />
              Critical: <span className={criticalAlerts > 0 ? 'text-red-400' : 'text-slate-400'}>{formatCount(criticalAlerts)}</span>
            </div>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto">{children}</main>
      </div>
    </div>
  )
}
