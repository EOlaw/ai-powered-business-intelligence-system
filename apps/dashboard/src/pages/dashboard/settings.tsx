/**
 * Dashboard — Settings Page (password change, org management)
 */

import React, { useState } from 'react';
import { Layout } from '../../components/Layout.js';
import { useAuth } from '../../hooks/useAuth.js';
import { authApi } from '../../services/api.js';

export default function SettingsPage() {
  const { accessToken } = useAuth();
  const [current, setCurrent]   = useState('');
  const [next, setNext]         = useState('');
  const [msg, setMsg]           = useState('');
  const [loading, setLoading]   = useState(false);

  async function changePassword(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setMsg('');
    try {
      await fetch(`${process.env['NEXT_PUBLIC_GATEWAY_URL'] ?? 'http://localhost:3000'}/users/me/password`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${accessToken}` },
        body:    JSON.stringify({ currentPassword: current, newPassword: next }),
      });
      setMsg('Password changed. You will be logged out of all other sessions.');
      setCurrent(''); setNext('');
    } catch (err: any) {
      setMsg(err.message ?? 'Failed to change password');
    } finally {
      setLoading(false);
    }
  }

  return (
    <Layout>
      <h1 className="text-2xl font-bold text-white mb-6">Settings</h1>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 max-w-md">
        <h2 className="text-white font-semibold mb-4">Change Password</h2>
        {msg && <p className="text-sm text-green-400 mb-4">{msg}</p>}
        <form onSubmit={changePassword} className="space-y-4">
          <div>
            <label className="block text-sm text-gray-400 mb-1">Current password</label>
            <input type="password" value={current} onChange={e => setCurrent(e.target.value)} required
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">New password</label>
            <input type="password" value={next} onChange={e => setNext(e.target.value)} required minLength={8}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
          <button type="submit" disabled={loading}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-sm font-medium px-5 py-2 rounded-lg">
            {loading ? 'Saving…' : 'Update password'}
          </button>
        </form>
      </div>
    </Layout>
  );
}
