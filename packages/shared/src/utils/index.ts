export function createRequestId(prefix = 'req'): string {
  const random = Math.random().toString(36).slice(2, 10);
  return `${prefix}_${Date.now().toString(36)}_${random}`;
}

export function assertNever(value: never): never {
  throw new Error(`Unhandled value: ${String(value)}`);
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

export function average(values: number[]): number {
  return values.length === 0 ? 0 : sum(values) / values.length;
}

export function percentile(values: number[], percentileRank: number): number {
  if (values.length === 0) {
    return 0;
  }

  const sorted = [...values].sort((a, b) => a - b);
  const index = clamp(Math.ceil((percentileRank / 100) * sorted.length) - 1, 0, sorted.length - 1);
  return sorted[index];
}

export function redactSecret(value: string, visible = 4): string {
  if (value.length <= visible) {
    return '*'.repeat(value.length);
  }

  return `${value.slice(0, visible)}${'*'.repeat(Math.max(value.length - visible, 0))}`;
}
