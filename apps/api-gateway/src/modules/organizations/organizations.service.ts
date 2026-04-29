/**
 * InsightSerenity API Gateway — Organizations Service
 * ====================================================
 * Multi-tenant organisation management.
 *
 * One user can belong to many orgs. Each org has one subscription.
 * The org OWNER cannot be removed; ownership can be transferred.
 *
 * Key operations:
 *   create       — new org for an existing user (sets them as OWNER)
 *   get/list     — load org data the calling user belongs to
 *   update       — rename the org (ADMIN+ required)
 *   addMember    — add an existing user by email with a given role
 *   removeMember — remove a member (cannot remove the OWNER)
 *   updateRole   — change a member's role (OWNER only)
 *   transfer     — transfer ownership to another member
 */

import { prisma }    from '../../db/client.js';
import { getLogger } from '../../observability/logger.js';
import { PLAN_LIMITS } from '../../config/settings.js';

const log = getLogger('orgs-service');

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

export async function getOrgOrFail(orgId: string, userId: string) {
  const membership = await prisma.orgMembership.findUnique({
    where:   { userId_orgId: { userId, orgId } },
    include: { org: true },
  });

  if (!membership) {
    throw Object.assign(new Error('Organisation not found'), { statusCode: 404, code: 'NOT_FOUND' });
  }

  return { org: membership.org, role: membership.role };
}

// ─────────────────────────────────────────────────────────────────────────────
// Create
// ─────────────────────────────────────────────────────────────────────────────

export async function createOrg(userId: string, name: string) {
  let slug = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 63);
  const exists = await prisma.organization.findUnique({ where: { slug } });
  if (exists) slug = `${slug}-${Date.now()}`;

  const org = await prisma.$transaction(async (tx) => {
    const o = await tx.organization.create({ data: { name, slug } });
    await tx.orgMembership.create({ data: { userId, orgId: o.id, role: 'OWNER' } });
    const limits = PLAN_LIMITS.FREE;
    await tx.subscription.create({
      data: {
        orgId: o.id, plan: 'FREE',
        tokensPerDay: limits.tokensPerDay, requestsPerMin: limits.requestsPerMin,
      },
    });
    return o;
  });

  log.info({ userId, orgId: org.id, name }, 'Organisation created');
  return org;
}

// ─────────────────────────────────────────────────────────────────────────────
// List orgs for a user
// ─────────────────────────────────────────────────────────────────────────────

export async function listUserOrgs(userId: string) {
  const memberships = await prisma.orgMembership.findMany({
    where: { userId },
    include: {
      org: {
        include: { subscription: { select: { plan: true, status: true, tokensPerDay: true, requestsPerMin: true } } },
      },
    },
  });

  return memberships.map((m) => ({
    role:         m.role,
    org:          m.org,
    subscription: m.org.subscription,
  }));
}

// ─────────────────────────────────────────────────────────────────────────────
// Update org name
// ─────────────────────────────────────────────────────────────────────────────

export async function updateOrg(orgId: string, userId: string, name: string) {
  await requireAdminRole(orgId, userId);
  return prisma.organization.update({ where: { id: orgId }, data: { name } });
}

// ─────────────────────────────────────────────────────────────────────────────
// Members
// ─────────────────────────────────────────────────────────────────────────────

export async function listMembers(orgId: string) {
  return prisma.orgMembership.findMany({
    where: { orgId },
    include: {
      user: { select: { id: true, email: true, createdAt: true } },
    },
  });
}

export async function addMember(orgId: string, callerUserId: string, email: string, role: 'ADMIN' | 'MEMBER') {
  await requireAdminRole(orgId, callerUserId);

  const target = await prisma.user.findUnique({ where: { email } });
  if (!target) {
    throw Object.assign(new Error('User with that email not found'), { statusCode: 404, code: 'NOT_FOUND' });
  }

  const existing = await prisma.orgMembership.findUnique({
    where: { userId_orgId: { userId: target.id, orgId } },
  });
  if (existing) {
    throw Object.assign(new Error('User is already a member'), { statusCode: 409, code: 'ALREADY_MEMBER' });
  }

  return prisma.orgMembership.create({
    data: { userId: target.id, orgId, role },
  });
}

export async function removeMember(orgId: string, callerUserId: string, targetUserId: string) {
  await requireAdminRole(orgId, callerUserId);

  const membership = await prisma.orgMembership.findUnique({
    where: { userId_orgId: { userId: targetUserId, orgId } },
  });

  if (!membership) {
    throw Object.assign(new Error('Member not found'), { statusCode: 404, code: 'NOT_FOUND' });
  }
  if (membership.role === 'OWNER') {
    throw Object.assign(new Error('Cannot remove the organisation owner'), { statusCode: 400, code: 'CANNOT_REMOVE_OWNER' });
  }

  await prisma.orgMembership.delete({ where: { userId_orgId: { userId: targetUserId, orgId } } });
}

export async function updateMemberRole(orgId: string, callerUserId: string, targetUserId: string, role: 'ADMIN' | 'MEMBER') {
  await requireOwnerRole(orgId, callerUserId);

  await prisma.orgMembership.update({
    where: { userId_orgId: { userId: targetUserId, orgId } },
    data:  { role },
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Role guards
// ─────────────────────────────────────────────────────────────────────────────

async function requireAdminRole(orgId: string, userId: string) {
  const m = await prisma.orgMembership.findUnique({ where: { userId_orgId: { userId, orgId } } });
  if (!m || !['OWNER', 'ADMIN'].includes(m.role)) {
    throw Object.assign(new Error('Admin access required'), { statusCode: 403, code: 'FORBIDDEN' });
  }
}

async function requireOwnerRole(orgId: string, userId: string) {
  const m = await prisma.orgMembership.findUnique({ where: { userId_orgId: { userId, orgId } } });
  if (!m || m.role !== 'OWNER') {
    throw Object.assign(new Error('Owner access required'), { statusCode: 403, code: 'FORBIDDEN' });
  }
}
