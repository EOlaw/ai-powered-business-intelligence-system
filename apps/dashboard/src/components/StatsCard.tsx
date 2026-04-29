/**
 * Dashboard — Stats Card component
 */

import React from 'react';
import clsx  from 'clsx';

interface StatsCardProps {
  title:   string;
  value:   string | number;
  sub?:    string;
  trend?:  'up' | 'down' | 'neutral';
  color?:  'indigo' | 'green' | 'yellow' | 'red';
}

export function StatsCard({ title, value, sub, trend, color = 'indigo' }: StatsCardProps) {
  const colorMap = {
    indigo: 'text-indigo-400',
    green:  'text-green-400',
    yellow: 'text-yellow-400',
    red:    'text-red-400',
  };

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <p className="text-sm text-gray-500 mb-1">{title}</p>
      <p className={clsx('text-3xl font-bold', colorMap[color])}>{value}</p>
      {sub && (
        <p className={clsx(
          'text-xs mt-1',
          trend === 'up'   ? 'text-green-400' :
          trend === 'down' ? 'text-red-400'   : 'text-gray-500',
        )}>
          {trend === 'up' ? '↑' : trend === 'down' ? '↓' : ''} {sub}
        </p>
      )}
    </div>
  );
}
