import {
  PIPELINE_STAGE_PATHS,
  type ApiCallOptions,
  type ApiResponse,
  type DashboardAnalytics,
  type DashboardOverview,
  type DashboardRecord,
  type DashboardSummary,
  type AdaptiveThresholdData,
  type ExplainabilityRecord,
  type ExplainabilitySummaryData,
  type FeedbackHistoryData,
  type FeedbackRecord,
  type FeedbackRequest,
  type FullPipelineOptions,
  type PipelineRunData,
  type PipelineStageName,
  type PipelineStatusData,
  type ReasoningSummaryData,
  type ReportCatalogData,
} from '../../types/Anomaly_Health_Monitering';
import { apiRequest } from './httpClient';

function post<T>(path: string, body?: unknown, options: ApiCallOptions = {}): Promise<ApiResponse<T>> {
  return apiRequest<T>(path, { method: 'POST', body, signal: options.signal });
}

function get<T>(path: string, options: ApiCallOptions = {}): Promise<ApiResponse<T>> {
  return apiRequest<T>(path, { signal: options.signal });
}

function queryString(values: FullPipelineOptions): string {
  const query = new URLSearchParams();

  Object.entries(values).forEach(([key, value]) => {
    if (value !== undefined) {
      query.set(key, String(value));
    }
  });

  const serialized = query.toString();
  return serialized ? `?${serialized}` : '';
}

export function getDashboardSummary(
  options?: ApiCallOptions,
): Promise<ApiResponse<DashboardSummary>> {
  return get('/dashboard/summary', options);
}

export function getLatestAllUnits(
  options?: ApiCallOptions,
): Promise<ApiResponse<DashboardRecord[]>> {
  return get('/dashboard/latest-all', options);
}

export function getDashboardOverview(
  options?: ApiCallOptions,
): Promise<ApiResponse<DashboardOverview>> {
  return get('/dashboard/overview', options);
}

export function getLatestUnit(
  unitId: number,
  options?: ApiCallOptions,
): Promise<ApiResponse<DashboardRecord>> {
  return post('/dashboard/latest-unit', { unit_id: unitId }, options);
}

export function getHealthTrend(
  unitId: number,
  options?: ApiCallOptions,
): Promise<ApiResponse<DashboardRecord[]>> {
  return post('/dashboard/health-trend', { unit_id: unitId }, options);
}

export function getUnitAnomalies(
  unitId: number,
  options?: ApiCallOptions,
): Promise<ApiResponse<DashboardRecord[]>> {
  return post('/dashboard/anomalies', { unit_id: unitId }, options);
}

export function getExplanation(
  unitId: number,
  cycle: number,
  options?: ApiCallOptions,
): Promise<ApiResponse<ExplainabilityRecord>> {
  return post('/dashboard/explanation', { unit_id: unitId, cycle }, options);
}

export function getConfidence(
  unitId: number,
  options?: ApiCallOptions,
): Promise<ApiResponse<DashboardRecord[]>> {
  return post('/dashboard/confidence', { unit_id: unitId }, options);
}

export function getDashboardAnalytics(
  options?: ApiCallOptions,
): Promise<ApiResponse<DashboardAnalytics>> {
  return get('/dashboard/analytics', options);
}

export function getAllAnomalies(
  options?: ApiCallOptions,
): Promise<ApiResponse<DashboardRecord[]>> {
  return get('/dashboard/anomalies-all', options);
}

export function getReasoning(
  options?: ApiCallOptions,
): Promise<ApiResponse<ReasoningSummaryData>> {
  return get('/dashboard/reasoning-summary', options);
}

export function getExplainability(
  options?: ApiCallOptions,
): Promise<ApiResponse<ExplainabilitySummaryData>> {
  return get('/dashboard/explainability-summary', options);
}

export function getFeedback(
  options?: ApiCallOptions,
): Promise<ApiResponse<FeedbackHistoryData>> {
  return get('/dashboard/feedback-history', options);
}

export function getThresholds(
  options?: ApiCallOptions,
): Promise<ApiResponse<AdaptiveThresholdData>> {
  return get('/dashboard/adaptive-thresholds', options);
}

export function getPipelineStatus(
  options?: ApiCallOptions,
): Promise<ApiResponse<PipelineStatusData>> {
  return get('/dashboard/pipeline-status', options);
}

export function getReports(
  options?: ApiCallOptions,
): Promise<ApiResponse<ReportCatalogData>> {
  return get('/dashboard/reports', options);
}

export function generateDashboard(
  options?: ApiCallOptions,
): Promise<ApiResponse<PipelineRunData>> {
  return post('/dashboard', undefined, options);
}

export function submitFeedback(
  request: FeedbackRequest,
  options?: ApiCallOptions,
): Promise<ApiResponse<FeedbackRecord | Record<string, unknown>>> {
  return post('/feedback', request, options);
}

export function uploadDataset(
  file: File,
  options: ApiCallOptions = {},
): Promise<ApiResponse<Record<string, unknown>>> {
  const formData = new FormData();
  formData.append('file', file);
  return apiRequest('/upload', { method: 'POST', body: formData, signal: options.signal });
}

export function runPipelineStage(
  stage: PipelineStageName,
  options?: ApiCallOptions,
): Promise<ApiResponse<PipelineRunData>> {
  return post(PIPELINE_STAGE_PATHS[stage], undefined, options);
}

export function evaluatePipeline(
  options?: ApiCallOptions,
): Promise<ApiResponse<PipelineRunData>> {
  return post('/evaluate', undefined, options);
}

export function runFullPipeline(
  pipelineOptions: FullPipelineOptions = {},
  options?: ApiCallOptions,
): Promise<ApiResponse<PipelineRunData>> {
  return post(`/full-pipeline${queryString(pipelineOptions)}`, undefined, options);
}

export const dashboardApi = {
  getDashboardSummary,
  getDashboardOverview,
  getLatestAllUnits,
  getLatestUnit,
  getHealthTrend,
  getUnitAnomalies,
  getExplanation,
  getConfidence,
  getDashboardAnalytics,
  getAllAnomalies,
  getReasoning,
  getExplainability,
  getFeedback,
  getThresholds,
  getPipelineStatus,
  getReports,
  generateDashboard,
  submitFeedback,
  uploadDataset,
  runPipelineStage,
  evaluatePipeline,
  runFullPipeline,
};
