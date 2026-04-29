import pino from 'pino';

export const logger = pino({
  level: process.env['LOG_LEVEL'] ?? 'info',
  base:  { service: 'worker', env: process.env['NODE_ENV'] ?? 'development' },
  timestamp: pino.stdTimeFunctions.isoTime,
  transport: process.env['LOG_PRETTY'] === 'true'
    ? { target: 'pino-pretty', options: { colorize: true, translateTime: 'SYS:HH:MM:ss' } }
    : undefined,
});

export function getLogger(module: string) {
  return logger.child({ module });
}
