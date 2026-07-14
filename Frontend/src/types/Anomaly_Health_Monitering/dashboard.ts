export type ApiStatus = 'success' | 'failed' | 'partial_failure' | 'not_found' | string;

export interface ApiResponse<T = unknown> {
  status: ApiStatus;
  message: string;
  output_file?: string | null;
  records_count?: number | null;
  metrics?: Record<string, unknown> | null;
  errors?: string[] | null;
  data?: T | null;
  [key: string]: unknown;
}

export interface ApiCallOptions {
  signal?: AbortSignal;
}

export interface UnitRequest {
  unit_id: number;
}

export interface UnitCycleRequest extends UnitRequest {
  cycle: number;
}

export type FeedbackLabel =
  | 'accepted_alert'
  | 'rejected_false_alarm'
  | 'missed_anomaly'
  | 'uncertain';

export interface FeedbackRequest extends UnitCycleRequest {
  context_id?: string | null;
  alert_level?: string | null;
  final_anomaly_score?: number | null;
  root_cause_pattern?: string | null;
  feedback_label: FeedbackLabel;
  operator_note?: string | null;
}

export interface AlertCounts {
  normal?: number;
  watch?: number;
  warning?: number;
  critical?: number;
  [key: string]: number | undefined;
}

export interface AnomalySummary {
  anomaly_records?: number;
  normal_records?: number;
  anomaly_detected?: boolean;
  anomaly_ratio?: number;
  [key: string]: unknown;
}

export interface HealthStateCounts {
  healthy?: number;
  degrading?: number;
  warning?: number;
  critical?: number;
  [key: string]: number | undefined;
}

export interface DashboardSummary {
  alert_counts?: AlertCounts;
  anomaly_summary?: AnomalySummary;
  health_state_counts?: HealthStateCounts;
  total_records?: number;
  unique_units?: number | null;
  average_health_index?: number | null;
  average_confidence_score?: number | null;
  average_uncertainty_score?: number | null;
  average_reliability_score?: number | null;
  [key: string]: unknown;
}

/**
 * A dashboard row is assembled from several pipeline output files. Fields are
 * optional because older generated datasets may not contain every stage yet.
 */
export interface DashboardRecord {
  unit_id?: number;
  cycle?: number;
  split?: string | null;
  context_id?: string | number | null;
  health_index?: number | null;
  remaining_health_percentage?: number | null;
  health_score?: number | null;
  health_state?: string | null;
  final_anomaly_score?: number | null;
  anomaly_detected?: boolean | number | null;
  anomaly_status?: string | null;
  alert_level?: string | null;
  severity?: string | null;
  early_warning_score?: number | null;
  confidence_score?: number | null;
  uncertainty_score?: number | null;
  reliability_score?: number | null;
  model_agreement_score?: number | null;
  root_cause_pattern?: string | null;
  inspection_focus?: string | null;
  primary_subsystem?: string | null;
  subsystem_explanation?: string | null;
  top_sensor_1?: string | null;
  top_sensor_2?: string | null;
  top_sensor_3?: string | null;
  contribution_1?: number | null;
  contribution_2?: number | null;
  contribution_3?: number | null;
  explanation_text?: string | null;
  feedback_label?: string | null;
  feedback_status?: string | null;
  [key: string]: unknown;
}

export interface DashboardAnalytics {
  summaries?: Record<string, unknown>;
  sources?: ArtifactMetadata[];
  health_distribution?: Record<string, number>;
  alert_distribution?: Record<string, number>;
  subsystem_distribution?: Record<string, number>;
  anomaly_trend?: Array<Record<string, unknown>>;
  health_trend?: Array<Record<string, unknown>>;
  top_sensors?: Array<Record<string, unknown>>;
  [key: string]: unknown;
}

export interface ArtifactMetadata {
  name?: string;
  path?: string;
  extension?: string;
  size_bytes?: number | null;
  updated_at?: string | null;
  records_count?: number | null;
  [key: string]: unknown;
}

export interface DashboardOverview {
  aggregates?: Record<string, unknown>;
  source?: ArtifactMetadata;
  summary?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface ReasoningRecord extends DashboardRecord {
  root_cause_confidence?: number | null;
  root_cause_summary?: string | null;
  recommended_action?: string | null;
}

export interface ExplainabilityRecord extends DashboardRecord {
  feature?: string | null;
  importance?: number | null;
  shap_value?: number | null;
  direction?: string | null;
}

export interface FeedbackRecord extends DashboardRecord {
  timestamp?: string | null;
  operator_note?: string | null;
  previous_threshold?: number | null;
  updated_threshold?: number | null;
}

export interface ThresholdRecord {
  name?: string;
  sensor?: string | null;
  alert_level?: string | null;
  previous_value?: number | null;
  current_value?: number | null;
  adjustment?: number | null;
  updated_at?: string | null;
  [key: string]: unknown;
}

export interface ReasoningSummaryData {
  reports?: ReportRecord[];
  [key: string]: unknown;
}

export interface ExplainabilitySummaryData {
  reports?: ReportRecord[];
  shap_rows?: ExplainabilityRecord[];
  shap_file?: ArtifactMetadata | null;
  shap_rows_truncated?: boolean;
  [key: string]: unknown;
}

export interface FeedbackHistoryData {
  feedback?: FeedbackRecord[];
  recent_alerts?: DashboardRecord[];
  recent_alerts_sampling?: string;
  [key: string]: unknown;
}

export interface AdaptiveThresholdData {
  metadata?: ArtifactMetadata;
  content?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface PipelineStageStatus {
  name?: string;
  stage?: string;
  status?: ApiStatus;
  message?: string;
  started_at?: string | null;
  completed_at?: string | null;
  duration_seconds?: number | null;
  output_file?: string | null;
  records_count?: number | null;
  [key: string]: unknown;
}

export interface PipelineStatusData {
  overall_status?: ApiStatus;
  success_count?: number;
  failed_count?: number;
  not_run_count?: number;
  status?: ApiStatus;
  stages?: PipelineStageStatus[];
  completed_stages?: PipelineStageStatus[];
  failed_stages?: PipelineStageStatus[];
  current_stage?: string | null;
  last_run_at?: string | null;
  [key: string]: unknown;
}

export interface ReportRecord {
  name?: string;
  title?: string;
  filename?: string;
  category?: string;
  status?: ApiStatus;
  updated_at?: string | null;
  size_bytes?: number | null;
  summary?: string | null;
  message?: string | null;
  records_count?: number | null;
  content?: Record<string, unknown> | null;
  content_status?: string | null;
  data?: unknown;
  metrics?: Record<string, unknown> | null;
  [key: string]: unknown;
}

export interface ReportCatalogData {
  reports?: ReportRecord[];
  output_files?: ArtifactMetadata[];
  metric_files?: ArtifactMetadata[];
  report_count?: number;
  output_file_count?: number;
  metric_file_count?: number;
  truncated?: boolean;
  [key: string]: unknown;
}

export interface FullPipelineOptions {
  include_shap?: boolean;
  include_evaluation?: boolean;
  include_dashboard?: boolean;
  include_feedback?: boolean;
  include_context_drift?: boolean;
  include_twin_comparison?: boolean;
}

export interface PipelineRunData {
  status?: ApiStatus;
  message?: string;
  completed_stage_count?: number;
  failed_stage_count?: number;
  completed_stages?: PipelineStageStatus[];
  failed_stages?: PipelineStageStatus[];
  dashboard_file?: string | null;
  [key: string]: unknown;
}

export const PIPELINE_STAGE_PATHS = {
  preprocessing: '/preprocess',
  contextModeling: '/context-modeling',
  digitalTwin: '/train-digital-twin',
  residualAnalysis: '/generate-residuals',
  anomalyDetection: '/detect-anomalies',
  healthIndex: '/generate-health-index',
  healthScore: '/generate-health-score',
  healthClassification: '/classify-health-state',
  reasoning: '/root-cause-analysis',
  explainability: '/explain',
  uncertainty: '/confidence',
  dashboard: '/dashboard',
} as const;

export type PipelineStageName = keyof typeof PIPELINE_STAGE_PATHS;
