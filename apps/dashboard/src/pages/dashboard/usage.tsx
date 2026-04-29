/**
 * Dashboard — Usage Analytics Page
 */

import React             from 'react';
import { useQuery }      from '@tanstack/react-query';
import { Layout }        from '../../components/Layout.js';
import { StatsCard }     from '../../components/StatsCard.js';
import { UsageChart }    from '../../components/UsageChart.js';
import { useAuth }       from '../../hooks/useAuth.js';
import { usageApi }      from '../../services/api.js';

export default function UsagePage() {
  const { accessToken, orgId } = useAuth();

  const { data: overview }  = useQuery({
    queryKey: ['usage-overview', orgId],
    queryFn:  () => usageApi.overview(orgId!, accessToken!),
    enabled:  Boolean(orgId && accessToken),
  });

  const { data: timeline }  = useQuery({
    queryKey: ['usage-timeline', orgId],
    queryFn:  () => usageApi.timeline(orgId!, accessToken!),
    enabled:  Boolean(orgId && accessToken),
  });

  const { data: recentReqs } = useQuery({
    queryKey: ['usage-requests', orgId],
    queryFn:  () => usageApi.requests(orgId!, accessToken!),
    enabled:  Boolean(orgId && accessToken),
  });

  const t = overview?.totals ?? {};

  return (
    <Layout>
      <h1 className="text-2xl font-bold text-white mb-6">Usage Analytics</h1>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatsCard title="Requests"        value={(t.requests ?? 0).toLocaleString()}    color="indigo" />
        <StatsCard title="Total Tokens"    value={(t.totalTokens ?? 0).toLocaleString()}  color="green" />
        <StatsCard title="Prompt Tokens"   value={(t.promptTokens ?? 0).toLocaleString()} color="yellow" />
        <StatsCard title="Avg Latency"     value={`${t.avgLatencyMs ?? 0} ms`}            color="indigo" />
      </div>

      {timeline && (
        <div className="mb-8">
          <UsageChart data={(timeline ?? []).map((d: any) => ({
            day: d.day, totalTokens: d.totalTokens, requests: d.requests,
          }))} />
        </div>
      )}

      {/* Recent requests table */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-gray-800">
          <h3 className="text-sm font-medium text-white">Recent Requests</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800">
                {['Endpoint', 'Tokens', 'Latency', 'Status', 'Time'].map(h => (
                  <th key={h} className="text-left px-5 py-3 text-xs text-gray-500 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(recentReqs?.data ?? []).map((r: any) => (
                <tr key={r.id} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                  <td className="px-5 py-3 text-gray-300 font-mono text-xs">{r.endpoint}</td>
                  <td className="px-5 py-3 text-gray-400">{r.totalTokens.toLocaleString()}</td>
                  <td className="px-5 py-3 text-gray-400">{r.latencyMs} ms</td>
                  <td className="px-5 py-3">
                    <span className={r.statusCode < 300 ? 'text-green-400' : 'text-red-400'}>
                      {r.statusCode}
                    </span>
                  </td>
                  <td className="px-5 py-3 text-gray-500 text-xs">
                    {new Date(r.createdAt).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Layout>
  );
}
