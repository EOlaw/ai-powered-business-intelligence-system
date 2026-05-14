/**
 * Dashboard — Main Layout with Sidebar
 */

import React from 'react';
import Link  from 'next/link';
import { useRouter } from 'next/router';
import { useAuth }   from '../hooks/useAuth.js';
import clsx          from 'clsx';

const NAV_ITEMS = [
  { href: '/ask',               label: 'Ask AI',    icon: 'AI' },
  { href: '/dashboard',         label: 'Overview',  icon: '⊞' },
  { href: '/dashboard/api-keys', label: 'API Keys',  icon: '🔑' },
  { href: '/dashboard/usage',   label: 'Usage',     icon: '📊' },
  { href: '/dashboard/billing', label: 'Billing',   icon: '💳' },
  { href: '/dashboard/settings', label: 'Settings', icon: '⚙' },
];

export function Layout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const { logout } = useAuth();

  return (
    <div className="flex h-screen bg-gray-950 text-gray-100">
      {/* Sidebar */}
      <aside className="w-64 flex flex-col bg-gray-900 border-r border-gray-800">
        {/* Logo */}
        <div className="px-6 py-5 border-b border-gray-800">
          <span className="text-lg font-bold text-indigo-400">InsightSerenity</span>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-4 py-4 space-y-1">
          {NAV_ITEMS.map(item => (
            <Link
              key={item.href}
              href={item.href}
              className={clsx(
                'flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                router.pathname === item.href
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800',
              )}
            >
              <span>{item.icon}</span>
              {item.label}
            </Link>
          ))}
        </nav>

        {/* Logout */}
        <div className="px-4 py-4 border-t border-gray-800">
          <button
            onClick={logout}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
          >
            <span>→</span> Sign out
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-6xl mx-auto px-8 py-8">
          {children}
        </div>
      </main>
    </div>
  );
}
