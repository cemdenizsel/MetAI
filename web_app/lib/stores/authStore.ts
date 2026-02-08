import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { User, AuthState } from '@/types/auth';
import * as authApi from '@/lib/api/auth';
import { setCookie, deleteCookie } from 'cookies-next';

// Helper to set auth token in both localStorage and cookie
const setAuthToken = (token: string) => {
  localStorage.setItem('auth_token', token);
  setCookie('auth_token', token, { maxAge: 60 * 60 * 24 * 7 }); // 7 days
};

// Helper to remove auth token from both localStorage and cookie
const removeAuthToken = () => {
  localStorage.removeItem('auth_token');
  deleteCookie('auth_token');
};

interface AuthActions {
  login: (email: string, password: string) => Promise<void>;
  register: (username: string, email: string, password: string) => Promise<void>;
  logout: () => void;
  /** Returns true if authenticated, false otherwise. Use to redirect immediately. */
  loadUser: () => Promise<boolean>;
  setLoading: (isLoading: boolean) => void;
}

type AuthStore = AuthState & AuthActions;

export const useAuthStore = create<AuthStore>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: true,

      setLoading: (isLoading: boolean) => set({ isLoading }),

      login: async (email: string, password: string) => {
        set({ isLoading: true });
        try {
          const response = await authApi.login({ email, password });
          const token = response.access_token;

          setAuthToken(token);

          const profile = await authApi.getProfile();
          const user: User = {
            id: profile.id,
            email: profile.email,
            name: profile.name,
            created_at: profile.created_at,
          };

          localStorage.setItem('user', JSON.stringify(user));

          set({
            user,
            token,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },

      register: async (username: string, email: string, password: string) => {
        set({ isLoading: true });
        try {
          const response = await authApi.register({ username, email, password });
          const token = response.access_token;

          setAuthToken(token);

          const profile = await authApi.getProfile();
          const user: User = {
            id: profile.id,
            email: profile.email,
            name: profile.name,
            created_at: profile.created_at,
          };

          localStorage.setItem('user', JSON.stringify(user));

          set({
            user,
            token,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },

      logout: () => {
        authApi.logout();
        removeAuthToken();
        localStorage.removeItem('user');
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          isLoading: false,
        });
      },

      loadUser: async (): Promise<boolean> => {
        const token = localStorage.getItem('auth_token');
        const userStr = localStorage.getItem('user');

        if (!token || !userStr) {
          removeAuthToken();
          set({ isLoading: false, isAuthenticated: false });
          return false;
        }

        // Check if token is expired before making API call
        try {
          // Decode JWT to check expiry (without verification)
          const payload = JSON.parse(atob(token.split('.')[1]));
          const expiry = payload.exp * 1000; // Convert to milliseconds

          if (Date.now() >= expiry) {
            // Token expired - clean up and return false
            console.log('Token expired, clearing auth state');
            removeAuthToken();
            localStorage.removeItem('user');
            set({
              user: null,
              token: null,
              isAuthenticated: false,
              isLoading: false,
            });
            return false;
          }
        } catch (err) {
          // If token decode fails, try API call anyway
          console.warn('Failed to decode token, will validate with API', err);
        }

        // Ensure cookie is synced with localStorage (only if not expired)
        setCookie('auth_token', token, { maxAge: 60 * 60 * 24 * 7 });

        try {
          const profile = await authApi.getProfile();
          const user: User = {
            id: profile.id,
            email: profile.email,
            name: profile.name,
            created_at: profile.created_at,
          };

          set({
            user,
            token,
            isAuthenticated: true,
            isLoading: false,
          });
          return true;
        } catch (error) {
          // API call failed (network error or token invalid)
          console.error('Failed to load user profile, clearing auth', error);
          removeAuthToken();
          localStorage.removeItem('user');
          set({
            user: null,
            token: null,
            isAuthenticated: false,
            isLoading: false,
          });
          return false;
        }
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);
