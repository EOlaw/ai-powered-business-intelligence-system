/**
 * Dashboard — Root redirect
 */

import { useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth }   from '../hooks/useAuth.js';

export default function IndexPage() {
  const { isLoggedIn } = useAuth();
  const router = useRouter();

  useEffect(() => {
    router.replace(isLoggedIn ? '/dashboard' : '/login');
  }, [isLoggedIn]);

  return null;
}
