/**
 * Dashboard — Auth Store (Zustand)
 * =================================
 * Persists tokens in localStorage. Provides helpers used by
 * API calls and route guards.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface AuthState {
  accessToken:  string | null;
  refreshToken: string | null;
  userId:       string | null;
  orgId:        string | null;
  setTokens:    (tokens: { accessToken: string; refreshToken: string; userId: string; orgId?: string }) => void;
  clearTokens:  () => void;
  isLoggedIn:   () => boolean;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      accessToken:  null,
      refreshToken: null,
      userId:       null,
      orgId:        null,

      setTokens: ({ accessToken, refreshToken, userId, orgId }) =>
        set({ accessToken, refreshToken, userId, orgId: orgId ?? null }),

      clearTokens: () =>
        set({ accessToken: null, refreshToken: null, userId: null, orgId: null }),

      isLoggedIn: () => Boolean(get().accessToken),
    }),
    { name: 'is-auth' },
  ),
);
