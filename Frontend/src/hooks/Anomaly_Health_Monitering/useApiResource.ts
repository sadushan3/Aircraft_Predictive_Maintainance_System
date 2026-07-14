import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type DependencyList,
} from 'react';

export interface ApiResourceState<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
  refetch: () => Promise<T | null>;
}

export interface ApiResourceOptions<T> {
  enabled?: boolean;
  initialData?: T | null;
  keepPreviousData?: boolean;
  onSuccess?: (data: T) => void;
  onError?: (error: Error) => void;
}

export interface ApiMutationState<TResult, TVariables> {
  data: TResult | null;
  loading: boolean;
  error: Error | null;
  mutate: (variables: TVariables) => Promise<TResult>;
  reset: () => void;
}

function asError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError';
}

export function useApiResource<T>(
  loader: (signal: AbortSignal) => Promise<T>,
  dependencies: DependencyList = [],
  options: ApiResourceOptions<T> = {},
): ApiResourceState<T> {
  const {
    enabled = true,
    initialData = null,
    keepPreviousData = false,
    onSuccess,
    onError,
  } = options;
  const [data, setData] = useState<T | null>(initialData);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState<Error | null>(null);
  const loaderRef = useRef(loader);
  const successRef = useRef(onSuccess);
  const errorRef = useRef(onError);
  const controllerRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef(0);

  loaderRef.current = loader;
  successRef.current = onSuccess;
  errorRef.current = onError;

  const execute = useCallback(async (): Promise<T | null> => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    const requestId = ++requestIdRef.current;

    setLoading(true);
    setError(null);
    if (!keepPreviousData) {
      setData(null);
    }

    try {
      const result = await loaderRef.current(controller.signal);

      if (!controller.signal.aborted && requestId === requestIdRef.current) {
        setData(result);
        successRef.current?.(result);
      }

      return controller.signal.aborted ? null : result;
    } catch (caught) {
      if (controller.signal.aborted || isAbortError(caught)) {
        return null;
      }

      const nextError = asError(caught);
      if (requestId === requestIdRef.current) {
        setError(nextError);
        errorRef.current?.(nextError);
      }
      return null;
    } finally {
      if (!controller.signal.aborted && requestId === requestIdRef.current) {
        setLoading(false);
      }
    }
  }, dependencies);

  useEffect(() => {
    if (!enabled) {
      controllerRef.current?.abort();
      setLoading(false);
      return undefined;
    }

    void execute();
    return () => controllerRef.current?.abort();
  }, [enabled, execute]);

  return { data, loading, error, refetch: execute };
}

export function useApiMutation<TResult, TVariables = void>(
  mutation: (variables: TVariables, signal: AbortSignal) => Promise<TResult>,
): ApiMutationState<TResult, TVariables> {
  const [data, setData] = useState<TResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const mutationRef = useRef(mutation);
  const controllerRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef(0);

  mutationRef.current = mutation;

  useEffect(() => () => controllerRef.current?.abort(), []);

  const mutate = useCallback(async (variables: TVariables): Promise<TResult> => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    const requestId = ++requestIdRef.current;

    setLoading(true);
    setError(null);

    try {
      const result = await mutationRef.current(variables, controller.signal);
      if (!controller.signal.aborted && requestId === requestIdRef.current) {
        setData(result);
      }
      return result;
    } catch (caught) {
      if (!controller.signal.aborted && !isAbortError(caught)) {
        const nextError = asError(caught);
        if (requestId === requestIdRef.current) {
          setError(nextError);
        }
      }
      throw caught;
    } finally {
      if (!controller.signal.aborted && requestId === requestIdRef.current) {
        setLoading(false);
      }
    }
  }, []);

  const reset = useCallback(() => {
    controllerRef.current?.abort();
    requestIdRef.current += 1;
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return { data, loading, error, mutate, reset };
}

