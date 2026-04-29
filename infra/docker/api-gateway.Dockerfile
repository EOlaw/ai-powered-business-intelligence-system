# InsightSerenity API Gateway — Production Dockerfile
# =====================================================
# Multi-stage build:
#   deps    — install only production node_modules
#   builder — compile TypeScript
#   runtime — minimal node:20-alpine with compiled JS only
#
# Usage:
#   docker build -f infra/docker/api-gateway.Dockerfile -t insightserenity/api-gateway:latest .
#   docker run -p 3000:3000 --env-file .env insightserenity/api-gateway:latest

# ── Stage 1: Install dependencies ─────────────────────────────────────────────
FROM node:20-alpine AS deps

RUN apk add --no-cache libc6-compat openssl

WORKDIR /app

COPY apps/api-gateway/package*.json ./
RUN npm install --omit=dev

# ── Stage 2: Build TypeScript ──────────────────────────────────────────────────
FROM node:20-alpine AS builder

RUN apk add --no-cache libc6-compat openssl

WORKDIR /app

# Install ALL deps (including devDependencies for tsc)
COPY apps/api-gateway/package*.json ./
RUN npm install

COPY apps/api-gateway/tsconfig.json ./
COPY apps/api-gateway/src           ./src
COPY apps/api-gateway/prisma        ./prisma

# Generate Prisma client
RUN npx prisma generate

# Compile TypeScript
RUN npm run build

# ── Stage 3: Runtime ───────────────────────────────────────────────────────────
FROM node:20-alpine AS runtime

RUN apk add --no-cache libc6-compat openssl curl \
 && addgroup --system --gid 1001 nodejs \
 && adduser  --system --uid 1001 gateway

WORKDIR /app

# Copy compiled output + prisma client + production node_modules
COPY --from=builder --chown=gateway:nodejs /app/dist           ./dist
COPY --from=builder --chown=gateway:nodejs /app/node_modules/.prisma ./node_modules/.prisma
COPY --from=deps    --chown=gateway:nodejs /app/node_modules   ./node_modules
COPY apps/api-gateway/package.json .
COPY apps/api-gateway/prisma       ./prisma

USER gateway

ENV NODE_ENV=production \
    PORT=3000

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

CMD ["node", "dist/server.js"]
