/**
 * Dashboard — Overview Page
 * ==========================
 * Shows total requests, tokens, avg latency, and the 30-day usage chart.
 */

import React             from 'react';
import { useQuery }      from '@tanstack/react-query';
import { Layout }        from '../../components/Layout.js';
import { StatsCard }     from '../../components/StatsCard.js';
import { UsageChart }    from '../../components/UsageChart.js';
import { useAuth }       from '../../hooks/useAuth.js';
import { usageApi }      from '../../services/api.js';

export default function DashboardPage() {
  const { accessToken, orgId, isLoggedIn } = useAuth();

  const { data: overview, isLoading: loadingOverview } = useQuery({
    queryKey: ['usage-overview', orgId],
    queryFn:  () => usageApi.overview(orgId!, accessToken!),
    enabled:  Boolean(orgId && accessToken),
  });

  const { data: timeline, isLoading: loadingTimeline } = useQuery({
    queryKey: ['usage-timeline', orgId],
    queryFn:  () => usageApi.timeline(orgId!, accessToken!),
    enabled:  Boolean(orgId && accessToken),
  });

  if (!isLoggedIn) return null;

  const totals = overview?.totals ?? {};

  return (
    <Layout>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white">Overview</h1>
        <p className="text-gray-500 text-sm mt-1">Last 30 days</p>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatsCard title="Total Requests"   value={loadingOverview ? '…' : (totals.requests ?? 0).toLocaleString()}      color="indigo" />
        <StatsCard title="Total Tokens"     value={loadingOverview ? '…' : (totals.totalTokens ?? 0).toLocaleString()}    color="green" />
        <StatsCard title="Prompt Tokens"    value={loadingOverview ? '…' : (totals.promptTokens ?? 0).toLocaleString()}   color="yellow" />
        <StatsCard title="Avg Latency"      value={loadingOverview ? '…' : `${totals.avgLatencyMs ?? 0} ms`}               color="indigo" />
      </div>

      {/* Usage chart */}
      {loadingTimeline ? (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 h-64 flex items-center justify-center">
          <span className="text-gray-500 text-sm">Loading chart…</span>
        </div>
      ) : (
        <UsageChart data={(timeline ?? []).map((d: any) => ({
          day:         d.day,
          totalTokens: d.totalTokens,
          requests:    d.requests,
        }))} />
      )}
    </Layout>
  );
}
