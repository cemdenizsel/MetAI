import { apiClient } from './client';
import type { AuthResponse, ProfileResponse, LoginCredentials, RegisterCredentials } from '@/types/auth';
import { deleteCookie } from 'cookies-next';

export async function login(credentials: LoginCredentials): Promise<AuthResponse> {
  const response = await fetch('/api/auth/login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      email: credentials.email,
      password: credentials.password,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    
    // Handle 422 validation errors with detailed messages
    if (response.status === 422) {
      if (Array.isArray(error.detail)) {
        // FastAPI validation error format
        const errorMessages = error.detail.map((err: any) => {
          const field = err.loc?.slice(1).join('.') || 'field';
          return `${field}: ${err.msg}`;
        }).join(', ');
        throw new Error(errorMessages || 'Validation failed');
      } else if (typeof error.detail === 'string') {
        throw new Error(error.detail);
      }
    }
    
    throw new Error(error.detail || error.message || 'Login failed');
  }

  return response.json();
}

export async function register(credentials: RegisterCredentials): Promise<AuthResponse> {
  const response = await fetch('/api/auth/register', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(credentials),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    
    // Handle 422 validation errors with detailed messages
    if (response.status === 422) {
      if (Array.isArray(error.detail)) {
        // FastAPI validation error format
        const errorMessages = error.detail.map((err: any) => {
          const field = err.loc?.slice(1).join('.') || 'field';
          return `${field}: ${err.msg}`;
        }).join(', ');
        throw new Error(errorMessages || 'Validation failed');
      } else if (typeof error.detail === 'string') {
        throw new Error(error.detail);
      }
    }
    
    throw new Error(error.detail || error.message || 'Registration failed');
  }

  return response.json();
}

export async function getProfile(): Promise<ProfileResponse> {
  return apiClient<ProfileResponse>('/api/auth/profile');
}

export async function logout(): Promise<void> {
  // Clear localStorage
  localStorage.removeItem('auth_token');
  localStorage.removeItem('user');
  localStorage.removeItem('auth-storage'); // Clear Zustand persisted state

  // Delete the auth cookie using cookies-next to ensure proper deletion
  deleteCookie('auth_token', { path: '/' });
}
