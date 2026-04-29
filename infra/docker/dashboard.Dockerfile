# InsightSerenity Dashboard — Dockerfile
# =========================================
# Next.js 14 admin and user dashboard. Uses standalone output mode so the
# runtime image only ships the minimal files Next.js needs to serve pages.
#
# Build arg:
#   NEXT_PUBLIC_GATEWAY_URL — public URL of the API Gateway (default: http://localhost:3000)
#
# Usage:
#   docker build -f infra/docker/dashboard.Dockerfile \
#     --build-arg NEXT_PUBLIC_GATEWAY_URL=http://localhost:3000 \
#     -t insightserenity/dashboard:latest .
#   docker run -p 3001:3001 insightserenity/dashboard:latest

# ── Stage 1: Install dependencies ─────────────────────────────────────────────
FROM node:20-alpine AS deps

RUN apk add --no-cache libc6-compat
WORKDIR /app

COPY apps/dashboard/package*.json ./
# SDK is a workspace dep; provide a stub so npm ci resolves it
RUN mkdir -p node_modules/@insightserenity/sdk \
 && echo '{"name":"@insightserenity/sdk","version":"1.0.0","main":"index.js"}' \
      > node_modules/@insightserenity/sdk/package.json \
 && echo 'module.exports = {};' > node_modules/@insightserenity/sdk/index.js
RUN npm install --omit=dev

# ── Stage 2: Build Next.js ─────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

ARG NEXT_PUBLIC_GATEWAY_URL=http://localhost:3000
ENV NEXT_PUBLIC_GATEWAY_URL=$NEXT_PUBLIC_GATEWAY_URL
ENV NEXT_TELEMETRY_DISABLED=1

COPY apps/dashboard/package*.json ./
RUN mkdir -p node_modules/@insightserenity/sdk \
 && echo '{"name":"@insightserenity/sdk","version":"1.0.0","main":"index.js"}' \
      > node_modules/@insightserenity/sdk/package.json \
 && echo 'module.exports = {};' > node_modules/@insightserenity/sdk/index.js
RUN npm install

COPY apps/dashboard/tsconfig.json      ./
COPY apps/dashboard/next.config.js     ./
COPY apps/dashboard/tailwind.config.js ./
COPY apps/dashboard/postcss.config.js  ./
COPY apps/dashboard/src                ./src

RUN npm run build

# ── Stage 3: Runtime ───────────────────────────────────────────────────────────
FROM node:20-alpine AS runtime

RUN apk add --no-cache curl \
 && addgroup --system --gid 1001 nodejs \
 && adduser  --system --uid 1001 nextjs

WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3001

# standalone output bundles everything needed
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static     ./.next/static

USER nextjs

EXPOSE 3001

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:3001/ || exit 1

CMD ["node", "server.js"]
