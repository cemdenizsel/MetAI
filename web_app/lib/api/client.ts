type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';

interface RequestConfig {
  method?: HttpMethod;
  body?: unknown;
  headers?: Record<string, string>;
  isFormData?: boolean;
}

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public data?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function getAuthToken(): Promise<string | null> {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem('auth_token');
}

export async function apiClient<T>(
  endpoint: string,
  config: RequestConfig = {}
): Promise<T> {
  const { method = 'GET', body, headers = {}, isFormData = false } = config;
  const token = await getAuthToken();

  const requestHeaders: Record<string, string> = {
    ...headers,
  };

  if (token) {
    requestHeaders['Authorization'] = `Bearer ${token}`;
  }

  if (!isFormData && body) {
    requestHeaders['Content-Type'] = 'application/json';
  }

  const requestConfig: RequestInit = {
    method,
    headers: requestHeaders,
  };

  if (body) {
    requestConfig.body = isFormData ? (body as FormData) : JSON.stringify(body);
  }

  const response = await fetch(endpoint, requestConfig);

  if (!response.ok) {
    let errorMessage = 'Request failed';
    let errorData: unknown;

    try {
      errorData = await response.json();
      errorMessage =
        (errorData as { detail?: string })?.detail ||
        (errorData as { message?: string })?.message ||
        errorMessage;
    } catch {
      errorMessage = response.statusText || errorMessage;
    }

    throw new ApiError(response.status, errorMessage, errorData);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
}

export { ApiError };
