/**
 * Dashboard — API Keys Page
 * ==========================
 * Create, revoke, rotate, and delete INSIGHTSERENITY_API_KEYs.
 * Raw key is shown in a modal on creation only — not stored in the UI.
 */

import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Layout }    from '../../components/Layout.js';
import { useAuth }   from '../../hooks/useAuth.js';
import { apiKeysApi } from '../../services/api.js';

export default function ApiKeysPage() {
  const { accessToken, orgId } = useAuth();
  const qc = useQueryClient();

  const [newKeyName, setNewKeyName] = useState('');
  const [createdKey, setCreatedKey]  = useState<string | null>(null);
  const [creating, setCreating]      = useState(false);

  const { data: keys = [], isLoading } = useQuery({
    queryKey: ['api-keys', orgId],
    queryFn:  () => apiKeysApi.list(orgId!, accessToken!),
    enabled:  Boolean(orgId && accessToken),
  });

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!newKeyName.trim()) return;
    setCreating(true);
    try {
      const result = await apiKeysApi.create(
        orgId!,
        { name: newKeyName.trim(), scopes: ['completions:create', 'chat:create', 'embeddings:create', 'agents:run'] },
        accessToken!,
      );
      setCreatedKey(result.rawKey);
      setNewKeyName('');
      await qc.invalidateQueries({ queryKey: ['api-keys'] });
    } finally {
      setCreating(false);
    }
  }

  async function handleRevoke(keyId: string) {
    if (!confirm('Revoke this API key? All requests using it will immediately fail.')) return;
    await apiKeysApi.revoke(orgId!, keyId, accessToken!);
    await qc.invalidateQueries({ queryKey: ['api-keys'] });
  }

  return (
    <Layout>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white">API Keys</h1>
        <p className="text-gray-500 text-sm mt-1">Keys starting with <code className="text-indigo-400">is_sk_</code></p>
      </div>

      {/* Create key form */}
      <form onSubmit={handleCreate} className="bg-gray-900 border border-gray-800 rounded-xl p-5 mb-6 flex gap-3">
        <input
          value={newKeyName} onChange={e => setNewKeyName(e.target.value)}
          placeholder="Key name, e.g. Production"
          className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
        <button type="submit" disabled={creating || !newKeyName.trim()}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-sm font-medium px-5 py-2 rounded-lg">
          {creating ? 'Creating…' : '+ Create key'}
        </button>
      </form>

      {/* New key display (shown once) */}
      {createdKey && (
        <div className="bg-green-900/20 border border-green-700 rounded-xl p-5 mb-6">
          <p className="text-green-400 text-sm font-medium mb-2">✓ Key created — copy it now, it won't be shown again</p>
          <code className="block bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-indigo-300 text-sm break-all">
            {createdKey}
          </code>
          <button onClick={() => { navigator.clipboard.writeText(createdKey); }} className="mt-2 text-xs text-gray-400 hover:text-white">
            Copy to clipboard
          </button>
        </div>
      )}

      {/* Keys list */}
      {isLoading ? (
        <p className="text-gray-500 text-sm">Loading keys…</p>
      ) : keys.length === 0 ? (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-8 text-center">
          <p className="text-gray-500">No API keys yet. Create one above.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {keys.map((key: any) => (
            <div key={key.id} className="bg-gray-900 border border-gray-800 rounded-xl px-5 py-4 flex items-center justify-between">
              <div>
                <p className="text-white font-medium text-sm">{key.name}</p>
                <p className="text-gray-500 text-xs mt-0.5">
                  <code className="text-gray-400">{key.keyPrefix}…</code>
                  {' · '}
                  {key.isActive ? <span className="text-green-400">Active</span> : <span className="text-red-400">Revoked</span>}
                  {key.lastUsedAt ? ` · Last used ${new Date(key.lastUsedAt).toLocaleDateString()}` : ' · Never used'}
                </p>
                <div className="flex gap-2 mt-1">
                  {key.scopes.map((s: string) => (
                    <span key={s} className="text-xs bg-gray-800 text-gray-400 px-2 py-0.5 rounded">{s}</span>
                  ))}
                </div>
              </div>
              {key.isActive && (
                <button onClick={() => handleRevoke(key.id)}
                  className="text-xs text-red-400 hover:text-red-300 border border-red-800 hover:border-red-600 px-3 py-1.5 rounded-lg transition-colors">
                  Revoke
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </Layout>
  );
}
