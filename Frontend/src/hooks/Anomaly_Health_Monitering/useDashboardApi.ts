import {
  getAllAnomalies,
  getConfidence,
  getDashboardAnalytics,
  getDashboardOverview,
  getDashboardSummary,
  getExplainability,
  getExplanation,
  getFeedback,
  getHealthTrend,
  getLatestAllUnits,
  getLatestUnit,
  getPipelineStatus,
  getReasoning,
  getReports,
  getThresholds,
  getUnitAnomalies,
} from '../../api/Anomaly_Health_Monitering';
import type {
  DashboardAnalytics,
  AdaptiveThresholdData,
  DashboardOverview,
  DashboardRecord,
  DashboardSummary,
  ExplainabilityRecord,
  ExplainabilitySummaryData,
  FeedbackHistoryData,
  PipelineStatusData,
  ReasoningSummaryData,
  ReportCatalogData,
} from '../../types/Anomaly_Health_Monitering';
import {
  useApiResource,
  type ApiResourceOptions,
  type ApiResourceState,
} from './useApiResource';

function dataOr<T>(data: T | null | undefined, fallback: T): T {
  return data ?? fallback;
}

export function useDashboardSummary(
  options?: ApiResourceOptions<DashboardSummary>,
): ApiResourceState<DashboardSummary> {
  return useApiResource(
    async (signal) => dataOr((await getDashboardSummary({ signal })).data, {}),
    [],
    options,
  );
}

export function useLatestAllUnits(
  options?: ApiResourceOptions<DashboardRecord[]>,
): ApiResourceState<DashboardRecord[]> {
  return useApiResource(
    async (signal) => dataOr((await getLatestAllUnits({ signal })).data, []),
    [],
    options,
  );
}

export function useDashboardOverview(
  options?: ApiResourceOptions<DashboardOverview>,
): ApiResourceState<DashboardOverview> {
  return useApiResource(
    async (signal) => dataOr((await getDashboardOverview({ signal })).data, {}),
    [],
    options,
  );
}

export function useLatestUnit(
  unitId: number | null | undefined,
  options: ApiResourceOptions<DashboardRecord> = {},
): ApiResourceState<DashboardRecord> {
  return useApiResource(
    async (signal) => dataOr((await getLatestUnit(unitId as number, { signal })).data, {}),
    [unitId],
    { ...options, enabled: options.enabled !== false && unitId != null },
  );
}

export function useHealthTrend(
  unitId: number | null | undefined,
  options: ApiResourceOptions<DashboardRecord[]> = {},
): ApiResourceState<DashboardRecord[]> {
  return useApiResource(
    async (signal) => dataOr((await getHealthTrend(unitId as number, { signal })).data, []),
    [unitId],
    { ...options, enabled: options.enabled !== false && unitId != null },
  );
}

export function useUnitAnomalies(
  unitId: number | null | undefined,
  options: ApiResourceOptions<DashboardRecord[]> = {},
): ApiResourceState<DashboardRecord[]> {
  return useApiResource(
    async (signal) => dataOr((await getUnitAnomalies(unitId as number, { signal })).data, []),
    [unitId],
    { ...options, enabled: options.enabled !== false && unitId != null },
  );
}

export function useExplanation(
  unitId: number | null | undefined,
  cycle: number | null | undefined,
  options: ApiResourceOptions<ExplainabilityRecord> = {},
): ApiResourceState<ExplainabilityRecord> {
  const enabled = options.enabled !== false && unitId != null && cycle != null;
  return useApiResource(
    async (signal) =>
      dataOr((await getExplanation(unitId as number, cycle as number, { signal })).data, {}),
    [unitId, cycle],
    { ...options, enabled },
  );
}

export function useConfidence(
  unitId: number | null | undefined,
  options: ApiResourceOptions<DashboardRecord[]> = {},
): ApiResourceState<DashboardRecord[]> {
  return useApiResource(
    async (signal) => dataOr((await getConfidence(unitId as number, { signal })).data, []),
    [unitId],
    { ...options, enabled: options.enabled !== false && unitId != null },
  );
}

export function useDashboardAnalytics(
  options?: ApiResourceOptions<DashboardAnalytics>,
): ApiResourceState<DashboardAnalytics> {
  return useApiResource(
    async (signal) => dataOr((await getDashboardAnalytics({ signal })).data, {}),
    [],
    options,
  );
}

export function useAllAnomalies(
  options?: ApiResourceOptions<DashboardRecord[]>,
): ApiResourceState<DashboardRecord[]> {
  return useApiResource(
    async (signal) => dataOr((await getAllAnomalies({ signal })).data, []),
    [],
    options,
  );
}

export function useReasoning(
  options?: ApiResourceOptions<ReasoningSummaryData>,
): ApiResourceState<ReasoningSummaryData> {
  return useApiResource(
    async (signal) => dataOr((await getReasoning({ signal })).data, {}),
    [],
    options,
  );
}

export function useExplainability(
  options?: ApiResourceOptions<ExplainabilitySummaryData>,
): ApiResourceState<ExplainabilitySummaryData> {
  return useApiResource(
    async (signal) => dataOr((await getExplainability({ signal })).data, {}),
    [],
    options,
  );
}

export function useFeedback(
  options?: ApiResourceOptions<FeedbackHistoryData>,
): ApiResourceState<FeedbackHistoryData> {
  return useApiResource(
    async (signal) => dataOr((await getFeedback({ signal })).data, {}),
    [],
    options,
  );
}

export function useThresholds(
  options?: ApiResourceOptions<AdaptiveThresholdData>,
): ApiResourceState<AdaptiveThresholdData> {
  return useApiResource(
    async (signal) => dataOr((await getThresholds({ signal })).data, {}),
    [],
    options,
  );
}

export function usePipelineStatus(
  options?: ApiResourceOptions<PipelineStatusData>,
): ApiResourceState<PipelineStatusData> {
  return useApiResource(
    async (signal) => dataOr((await getPipelineStatus({ signal })).data, {}),
    [],
    options,
  );
}

export function useReports(
  options?: ApiResourceOptions<ReportCatalogData>,
): ApiResourceState<ReportCatalogData> {
  return useApiResource(
    async (signal) => dataOr((await getReports({ signal })).data, {}),
    [],
    options,
  );
}
