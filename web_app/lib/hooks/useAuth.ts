'use client';

import { useEffect } from 'react';
import { useAuthStore } from '@/lib/stores/authStore';

export function useAuth() {
  const store = useAuthStore();

  useEffect(() => {
    store.loadUser();
  }, []);

  return {
    user: store.user,
    isAuthenticated: store.isAuthenticated,
    isLoading: store.isLoading,
    login: store.login,
    register: store.register,
    logout: store.logout,
  };
}
