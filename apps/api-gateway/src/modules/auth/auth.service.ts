/**
 * InsightSerenity API Gateway — Auth Service
 * ==========================================
 * Business logic for user registration, login, token refresh, and logout.
 *
 * Registration flow:
 *   1. Validate email uniqueness
 *   2. Hash password with bcrypt
 *   3. Create User in DB
 *   4. Create Organization with the user's provided name
 *   5. Create OrgMembership(OWNER) linking user → org
 *   6. Create a FREE Subscription for the org
 *   7. Return token pair
 *
 * Login flow:
 *   1. Look up User by email
 *   2. Verify password hash (constant-time)
 *   3. Issue access + refresh token pair
 *   4. Store refresh token's jti in Session table
 *
 * Refresh flow:
 *   1. Verify refresh token signature and expiry
 *   2. Look up Session by jti — confirms token hasn't been revoked
 *   3. Issue new access token
 *
 * Logout flow:
 *   1. Delete Session by jti — permanently revokes the refresh token
 */

import { prisma }           from '../../db/client.js';
import { hashPassword, verifyPassword } from '../../security/crypto.js';
import { getLogger }        from '../../observability/logger.js';
import { PLAN_LIMITS }      from '../../config/settings.js';
import {
  issueTokenPair,
  issueAccessToken,
  verifyRefreshToken,
  refreshTokenExpiresAt,
  type TokenPair,
} from './strategies/jwt.strategy.js';

const log = getLogger('auth-service');

// ─────────────────────────────────────────────────────────────────────────────
// Register
// ─────────────────────────────────────────────────────────────────────────────

export async function register(
  email:   string,
  password: string,
  orgName:  string,
): Promise<{ tokens: TokenPair; userId: string; orgId: string }> {
  const existing = await prisma.user.findUnique({ where: { email } });
  if (existing) {
    throw Object.assign(new Error('Email already registered'), { statusCode: 409, code: 'EMAIL_TAKEN' });
  }

  const slug         = slugify(orgName);
  const slugExists   = await prisma.organization.findUnique({ where: { slug } });
  const finalSlug    = slugExists ? `${slug}-${Date.now()}` : slug;
  const passwordHash = await hashPassword(password);

  // Transactionally create User + Org + Membership + Subscription
  const [user, org] = await prisma.$transaction(async (tx) => {
    const u = await tx.user.create({
      data: { email, passwordHash },
    });

    const o = await tx.organization.create({
      data: { name: orgName, slug: finalSlug },
    });

    await tx.orgMembership.create({
      data: { userId: u.id, orgId: o.id, role: 'OWNER' },
    });

    const freeLimits = PLAN_LIMITS.FREE;
    await tx.subscription.create({
      data: {
        orgId:           o.id,
        plan:            'FREE',
        tokensPerDay:    freeLimits.tokensPerDay,
        requestsPerMin:  freeLimits.requestsPerMin,
      },
    });

    return [u, o];
  });

  const { tokens, jti } = issueTokenPair(user.id, user.email, 'USER');

  await prisma.session.create({
    data: {
      userId:    user.id,
      jti,
      expiresAt: refreshTokenExpiresAt(),
    },
  });

  log.info({ userId: user.id, orgId: org.id }, 'User registered');
  return { tokens, userId: user.id, orgId: org.id };
}

// ─────────────────────────────────────────────────────────────────────────────
// Login
// ─────────────────────────────────────────────────────────────────────────────

export async function login(
  email:    string,
  password: string,
): Promise<{ tokens: TokenPair; userId: string }> {
  const user = await prisma.user.findUnique({ where: { email } });

  // Constant-time: always verify the hash even if user doesn't exist
  const dummyHash = '$2a$12$invalidhashinvalidhashinvalidhash';
  const valid     = user
    ? await verifyPassword(password, user.passwordHash)
    : await verifyPassword(password, dummyHash).then(() => false);

  if (!user || !valid) {
    throw Object.assign(new Error('Invalid email or password'), { statusCode: 401, code: 'INVALID_CREDENTIALS' });
  }

  const { tokens, jti } = issueTokenPair(user.id, user.email, user.role as 'USER' | 'ADMIN');

  await prisma.session.create({
    data: {
      userId:    user.id,
      jti,
      expiresAt: refreshTokenExpiresAt(),
    },
  });

  log.info({ userId: user.id }, 'User logged in');
  return { tokens, userId: user.id };
}

// ─────────────────────────────────────────────────────────────────────────────
// Refresh
// ─────────────────────────────────────────────────────────────────────────────

export async function refresh(
  refreshToken: string,
): Promise<{ accessToken: string; expiresIn: number }> {
  let payload: { sub: string; jti: string };

  try {
    payload = verifyRefreshToken(refreshToken) as { sub: string; jti: string };
  } catch {
    throw Object.assign(new Error('Invalid or expired refresh token'), { statusCode: 401, code: 'INVALID_TOKEN' });
  }

  const session = await prisma.session.findUnique({ where: { jti: payload.jti } });

  if (!session || session.expiresAt < new Date()) {
    throw Object.assign(new Error('Session expired or revoked'), { statusCode: 401, code: 'SESSION_EXPIRED' });
  }

  const user = await prisma.user.findUnique({ where: { id: payload.sub } });
  if (!user) {
    throw Object.assign(new Error('User not found'), { statusCode: 401, code: 'USER_NOT_FOUND' });
  }

  return issueAccessToken(user.id, user.email, user.role as 'USER' | 'ADMIN');
}

// ─────────────────────────────────────────────────────────────────────────────
// Logout
// ─────────────────────────────────────────────────────────────────────────────

export async function logout(jti: string): Promise<void> {
  await prisma.session.deleteMany({ where: { jti } });
  log.info({ jti }, 'Session revoked');
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function slugify(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 63);
}
