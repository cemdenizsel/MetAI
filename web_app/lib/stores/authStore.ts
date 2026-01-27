import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { User, AuthState } from '@/types/auth';
import * as authApi from '@/lib/api/auth';

interface AuthActions {
  login: (email: string, password: string) => Promise<void>;
  register: (username: string, email: string, password: string) => Promise<void>;
  logout: () => void;
  loadUser: () => Promise<void>;
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

          localStorage.setItem('auth_token', token);

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

          localStorage.setItem('auth_token', token);

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
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          isLoading: false,
        });
      },

      loadUser: async () => {
        const token = localStorage.getItem('auth_token');
        const userStr = localStorage.getItem('user');

        if (!token || !userStr) {
          set({ isLoading: false, isAuthenticated: false });
          return;
        }

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
        } catch {
          localStorage.removeItem('auth_token');
          localStorage.removeItem('user');
          set({
            user: null,
            token: null,
            isAuthenticated: false,
            isLoading: false,
          });
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
