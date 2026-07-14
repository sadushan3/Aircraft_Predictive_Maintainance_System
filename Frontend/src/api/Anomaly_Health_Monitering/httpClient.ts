import type { ApiCallOptions, ApiResponse } from '../../types/Anomaly_Health_Monitering';

const DEFAULT_API_BASE_URL = '/api/anomaly-health-monitoring';

type Environment = ImportMeta & {
  env?: {
    VITE_API_BASE_URL?: string;
  };
};

const configuredBaseUrl = (import.meta as Environment).env?.VITE_API_BASE_URL?.trim();

/**
 * In development, `/api` is forwarded to FastAPI by Vite. In production,
 * VITE_API_BASE_URL can be an absolute URL that includes the backend router
 * prefix, for example `https://api.example.com/anomaly-health-monitoring`.
 */
export const API_BASE_URL = (configuredBaseUrl || DEFAULT_API_BASE_URL).replace(/\/+$/, '');

export class ApiError<T = unknown> extends Error {
  readonly statusCode: number | null;
  readonly response: ApiResponse<T> | null;

  constructor(
    message: string,
    options: { statusCode?: number | null; response?: ApiResponse<T> | null } = {},
  ) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = options.statusCode ?? null;
    this.response = options.response ?? null;
  }
}

interface RequestConfig extends ApiCallOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  body?: unknown;
  headers?: HeadersInit;
}

function endpointUrl(path: string): string {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE_URL}${normalizedPath}`;
}

async function readResponse<T>(response: Response): Promise<ApiResponse<T>> {
  const contentType = response.headers.get('content-type') ?? '';

  if (!contentType.includes('application/json')) {
    const message = (await response.text()).trim();
    throw new ApiError(message || `Request failed with HTTP ${response.status}.`, {
      statusCode: response.status,
    });
  }

  const payload = (await response.json()) as ApiResponse<T>;

  if (!response.ok) {
    throw new ApiError(payload.message || `Request failed with HTTP ${response.status}.`, {
      statusCode: response.status,
      response: payload,
    });
  }

  if (payload.status?.toLowerCase() === 'failed') {
    throw new ApiError(payload.message || 'The backend operation failed.', {
      statusCode: response.status,
      response: payload,
    });
  }

  return payload;
}

export async function apiRequest<T>(
  path: string,
  { method = 'GET', body, signal, headers }: RequestConfig = {},
): Promise<ApiResponse<T>> {
  const isFormData = typeof FormData !== 'undefined' && body instanceof FormData;
  const requestHeaders = new Headers(headers);

  requestHeaders.set('Accept', 'application/json');
  if (body !== undefined && !isFormData && !requestHeaders.has('Content-Type')) {
    requestHeaders.set('Content-Type', 'application/json');
  }

  const response = await fetch(endpointUrl(path), {
    method,
    headers: requestHeaders,
    signal,
    body:
      body === undefined
        ? undefined
        : isFormData
          ? body
          : JSON.stringify(body),
  });

  return readResponse<T>(response);
}

export function responseData<T>(response: ApiResponse<T>): T | null {
  return response.data ?? null;
}

