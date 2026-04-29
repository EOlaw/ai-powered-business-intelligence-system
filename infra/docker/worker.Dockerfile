# InsightSerenity Worker — Dockerfile
# =====================================
# Node.js 20 service that runs BullMQ pipeline workers and cron schedules.
# Stages:
#   deps    — production node_modules only
#   builder — compile TypeScript
#   runtime — minimal image with compiled JS
#
# Usage:
#   docker build -f infra/docker/worker.Dockerfile -t insightserenity/worker:latest .
#   docker run --env REDIS_URL=redis://redis:6379 insightserenity/worker:latest

# ── Stage 1: Production dependencies ──────────────────────────────────────────
FROM node:20-alpine AS deps

RUN apk add --no-cache libc6-compat
WORKDIR /app
COPY apps/worker/package*.json ./
RUN npm install --omit=dev

# ── Stage 2: Build TypeScript ──────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app
COPY apps/worker/package*.json ./
RUN npm install

COPY apps/worker/tsconfig.json ./
COPY apps/worker/src           ./src
RUN npm run build

# ── Stage 3: Runtime ───────────────────────────────────────────────────────────
FROM node:20-alpine AS runtime

RUN apk add --no-cache libc6-compat curl \
 && addgroup --system --gid 1001 nodejs \
 && adduser  --system --uid 1001 worker

WORKDIR /app

COPY --from=builder --chown=worker:nodejs /app/dist         ./dist
COPY --from=deps    --chown=worker:nodejs /app/node_modules ./node_modules
COPY apps/worker/package.json .

USER worker

ENV NODE_ENV=production

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD node -e "require('ioredis'); process.exit(0)" || exit 1

CMD ["node", "dist/main.js"]
