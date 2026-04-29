/**
 * Dashboard — Billing & Plan Page
 */

import React              from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Layout }         from '../../components/Layout.js';
import { useAuth }        from '../../hooks/useAuth.js';
import { billingApi }     from '../../services/api.js';
import clsx               from 'clsx';

const PLANS = [
  { key: 'FREE',       label: 'Free',       price: '$0/mo',  tokens: '1K tokens/day',    rpm: '10 req/min' },
  { key: 'STARTER',    label: 'Starter',    price: '$29/mo', tokens: '100K tokens/day',  rpm: '60 req/min' },
  { key: 'PRO',        label: 'Pro',        price: '$99/mo', tokens: '1M tokens/day',    rpm: '300 req/min' },
  { key: 'ENTERPRISE', label: 'Enterprise', price: 'Custom', tokens: 'Unlimited',        rpm: '1000 req/min' },
];

export default function BillingPage() {
  const { accessToken, orgId } = useAuth();
  const qc = useQueryClient();

  const { data: sub } = useQuery({
    queryKey: ['subscription', orgId],
    queryFn:  () => billingApi.subscription(orgId!, accessToken!),
    enabled:  Boolean(orgId && accessToken),
  });

  const changePlan = useMutation({
    mutationFn: (plan: string) => billingApi.changePlan(orgId!, plan, accessToken!),
    onSuccess:  () => qc.invalidateQueries({ queryKey: ['subscription'] }),
  });

  return (
    <Layout>
      <h1 className="text-2xl font-bold text-white mb-2">Billing</h1>
      <p className="text-gray-500 text-sm mb-8">
        Current plan: <span className="text-indigo-400 font-medium">{sub?.plan ?? '…'}</span>
        {' · '}Status: <span className="text-green-400">{sub?.status ?? '…'}</span>
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {PLANS.map(plan => {
          const isCurrent = sub?.plan === plan.key;
          return (
            <div key={plan.key} className={clsx(
              'bg-gray-900 border rounded-xl p-5 flex flex-col',
              isCurrent ? 'border-indigo-500' : 'border-gray-800',
            )}>
              <div className="flex items-start justify-between mb-3">
                <div>
                  <p className="text-white font-semibold">{plan.label}</p>
                  <p className="text-indigo-400 text-lg font-bold mt-0.5">{plan.price}</p>
                </div>
                {isCurrent && (
                  <span className="text-xs bg-indigo-600 text-white px-2 py-1 rounded-full">Current</span>
                )}
              </div>
              <ul className="text-sm text-gray-400 space-y-1 flex-1 mb-4">
                <li>✓ {plan.tokens}</li>
                <li>✓ {plan.rpm}</li>
              </ul>
              {!isCurrent && (
                <button
                  onClick={() => changePlan.mutate(plan.key)}
                  disabled={changePlan.isPending}
                  className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-sm font-medium py-2 rounded-lg"
                >
                  Switch to {plan.label}
                </button>
              )}
            </div>
          );
        })}
      </div>
    </Layout>
  );
}
