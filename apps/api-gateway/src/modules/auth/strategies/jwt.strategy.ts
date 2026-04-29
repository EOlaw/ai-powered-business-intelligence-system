/**
 * InsightSerenity API Gateway — JWT Token Strategy
 * =================================================
 * Centralises all JWT signing and verification logic.
 *
 * Two tokens are issued on login:
 *   accessToken   — short-lived (15m), stateless, verifies identity on every request
 *   refreshToken  — long-lived (7d), stored in DB (Session table), used only to
 *                   issue new access tokens; invalidated on logout
 *
 * The jti (JWT ID) claim in the refresh token is the primary key of the
 * Session row. Deleting that row is all it takes to revoke the token.
 *
 * Access tokens are NOT tracked in the DB — they expire naturally.
 * For immediate access token revocation (e.g. security incident), rotate
 * JWT_ACCESS_SECRET — this invalidates ALL outstanding access tokens at once.
 */

import crypto from 'node:crypto';
import jwt    from 'jsonwebtoken';
import { config }     from '../../../config/settings.js';
import type { JwtUser } from '../../../types/index.js';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface TokenPair {
  accessToken:  string;
  refreshToken: string;
  expiresIn:    number;   // Seconds until accessToken expires
}

export interface RefreshPayload {
  sub: string;   // User ID
  jti: string;   // Session ID (matches Session.jti in DB)
}

// ─────────────────────────────────────────────────────────────────────────────
// Issue tokens
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Issue an access + refresh token pair for a user.
 *
 * @param userId  User.id (cuid)
 * @param email   User.email
 * @param role    UserRole ('USER' | 'ADMIN')
 * @returns TokenPair and the jti to store in the Session table
 */
export function issueTokenPair(
  userId: string,
  email:  string,
  role:   'USER' | 'ADMIN',
): { tokens: TokenPair; jti: string } {
  const jti = crypto.randomBytes(16).toString('hex');

  const expiresIn = parseExpiresIn(config.JWT_ACCESS_EXPIRES_IN);

  const accessPayload: JwtUser = { sub: userId, email, role, jti: '' };
  const accessToken = jwt.sign(accessPayload, config.JWT_ACCESS_SECRET, {
    expiresIn,
  });

  const refreshPayload: RefreshPayload = { sub: userId, jti };
  const refreshToken = jwt.sign(refreshPayload, config.JWT_REFRESH_SECRET, {
    expiresIn: parseExpiresIn(config.JWT_REFRESH_EXPIRES_IN),
  });

  return { tokens: { accessToken, refreshToken, expiresIn }, jti };
}

/**
 * Issue a new access token for an existing, validated refresh token.
 */
export function issueAccessToken(
  userId: string,
  email:  string,
  role:   'USER' | 'ADMIN',
): { accessToken: string; expiresIn: number } {
  const expiresIn = parseExpiresIn(config.JWT_ACCESS_EXPIRES_IN);
  const accessPayload: JwtUser = { sub: userId, email, role, jti: '' };
  const accessToken = jwt.sign(accessPayload, config.JWT_ACCESS_SECRET, { expiresIn });
  return { accessToken, expiresIn };
}

// ─────────────────────────────────────────────────────────────────────────────
// Verify tokens
// ─────────────────────────────────────────────────────────────────────────────

/** Verify and decode an access token. Throws if invalid or expired. */
export function verifyAccessToken(token: string): JwtUser {
  return jwt.verify(token, config.JWT_ACCESS_SECRET) as JwtUser;
}

/** Verify and decode a refresh token. Throws if invalid or expired. */
export function verifyRefreshToken(token: string): RefreshPayload {
  return jwt.verify(token, config.JWT_REFRESH_SECRET) as RefreshPayload;
}

/** Refresh token expiry as a JavaScript Date (for DB storage). */
export function refreshTokenExpiresAt(): Date {
  const seconds = parseExpiresIn(config.JWT_REFRESH_EXPIRES_IN);
  return new Date(Date.now() + seconds * 1000);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Parse "15m", "7d", "3600" → seconds. */
function parseExpiresIn(expiresIn: string): number {
  const n = parseInt(expiresIn, 10);
  if (expiresIn.endsWith('m')) return n * 60;
  if (expiresIn.endsWith('h')) return n * 3_600;
  if (expiresIn.endsWith('d')) return n * 86_400;
  return n; // assume seconds
}
