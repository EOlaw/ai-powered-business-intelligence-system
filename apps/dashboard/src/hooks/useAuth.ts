/**
 * Dashboard — useAuth hook
 * =========================
 * Login, register, and logout helpers with query-state binding.
 */

import { useRouter }    from 'next/router';
import { useAuthStore } from '../store/auth.js';
import { authApi }      from '../services/api.js';

export function useAuth() {
  const store  = useAuthStore();
  const router = useRouter();

  async function login(email: string, password: string) {
    const tokens = await authApi.login(email, password);
    store.setTokens(tokens);
    await router.push('/dashboard');
  }

  async function register(email: string, password: string, orgName: string) {
    const tokens = await authApi.register(email, password, orgName);
    store.setTokens(tokens);
    await router.push('/dashboard');
  }

  async function logout() {
    store.clearTokens();
    await router.push('/login');
  }

  return {
    isLoggedIn:  store.isLoggedIn(),
    accessToken: store.accessToken,
    orgId:       store.orgId,
    userId:      store.userId,
    login,
    register,
    logout,
  };
}
