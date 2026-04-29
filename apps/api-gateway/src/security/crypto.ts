/**
 * InsightSerenity API Gateway — Cryptographic Utilities
 * =======================================================
 * All cryptographic operations for the gateway in one place:
 *
 *   API Key generation:
 *     Format: is_sk_<base62-random>
 *     - "is_sk_" prefix  → identifies InsightSerenity Secret Keys
 *     - base62 random    → URL-safe, no ambiguous chars, case-sensitive
 *
 *   API Key storage:
 *     Raw key is shown ONCE on creation, never stored.
 *     SHA-256 hash stored in DB for O(1) lookup on each request.
 *     Constant-time comparison used to prevent timing attacks.
 *
 *   Password hashing:
 *     bcrypt with configurable rounds (default: 12).
 *
 *   Webhook signing:
 *     HMAC-SHA256 of the request body using the webhook secret.
 */

import crypto from 'node:crypto';
import bcrypt  from 'bcryptjs';
import { config } from '../config/settings.js';

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const BASE62_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
const KEY_PREFIX   = 'is_sk_';

// ─────────────────────────────────────────────────────────────────────────────
// API Key operations
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Generate a new INSIGHTSERENITY_API_KEY.
 *
 * Returns the raw key (show once to the user) and its SHA-256 hash
 * (store in the database — never store the raw key).
 *
 * @param byteLength Number of random bytes before base62 encoding. Default: config value.
 */
export function generateApiKey(byteLength = config.API_KEY_BYTES): {
  rawKey:    string;
  keyHash:   string;
  keyPrefix: string;
} {
  // Generate cryptographically-random bytes
  const randomBytes = crypto.randomBytes(byteLength);

  // Convert to base62 for URL-safe, readable keys
  const base62 = bytesToBase62(randomBytes);

  const rawKey    = `${KEY_PREFIX}${base62}`;
  const keyHash   = hashApiKey(rawKey);
  const keyPrefix = rawKey.slice(0, KEY_PREFIX.length + 8); // "is_sk_Xy1z2w3a"

  return { rawKey, keyHash, keyPrefix };
}

/**
 * SHA-256 hash an API key for database storage.
 * Fast (no salt needed — keys are already high-entropy random strings).
 */
export function hashApiKey(rawKey: string): string {
  return crypto.createHash('sha256').update(rawKey).digest('hex');
}

/**
 * Constant-time comparison of two API key hashes.
 * Prevents timing attacks that could reveal partial hash matches.
 */
export function compareApiKeyHash(candidateKey: string, storedHash: string): boolean {
  const candidateHash = hashApiKey(candidateKey);
  // Both hashes are fixed-length hex strings — timingSafeEqual requires equal lengths
  try {
    return crypto.timingSafeEqual(
      Buffer.from(candidateHash, 'hex'),
      Buffer.from(storedHash,    'hex'),
    );
  } catch {
    return false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Password operations
// ─────────────────────────────────────────────────────────────────────────────

/** Hash a plaintext password using bcrypt. */
export async function hashPassword(password: string): Promise<string> {
  return bcrypt.hash(password, config.BCRYPT_ROUNDS);
}

/** Verify a plaintext password against a bcrypt hash. */
export async function verifyPassword(password: string, hash: string): Promise<boolean> {
  return bcrypt.compare(password, hash);
}

// ─────────────────────────────────────────────────────────────────────────────
// Webhook signing
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Generate an HMAC-SHA256 signature for a webhook payload.
 *
 * The signature header format is:
 *   X-InsightSerenity-Signature: sha256=<hex-digest>
 *
 * Receivers verify by computing the same HMAC and doing a constant-time compare.
 */
export function signWebhookPayload(payload: string, secret: string): string {
  const hmac = crypto.createHmac('sha256', secret);
  hmac.update(payload);
  return `sha256=${hmac.digest('hex')}`;
}

/** Verify a webhook signature header against the payload and secret. */
export function verifyWebhookSignature(
  payload:   string,
  secret:    string,
  signature: string,
): boolean {
  const expected = signWebhookPayload(payload, secret);
  try {
    return crypto.timingSafeEqual(
      Buffer.from(expected,  'utf8'),
      Buffer.from(signature, 'utf8'),
    );
  } catch {
    return false;
  }
}

/** Generate a random webhook secret (32 random bytes, hex-encoded). */
export function generateWebhookSecret(): string {
  return crypto.randomBytes(32).toString('hex');
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Convert a Buffer of random bytes to a base62 string.
 * Uses rejection sampling to ensure uniform distribution.
 */
function bytesToBase62(bytes: Buffer): string {
  let result = '';
  for (const byte of bytes) {
    // Reject bytes >= 248 to avoid modulo bias (248 = 4 * 62)
    if (byte < 248) {
      result += BASE62_CHARS[byte % 62];
    }
  }
  // If we rejected too many bytes, pad with fresh random bytes
  while (result.length < bytes.length) {
    const extra = crypto.randomBytes(1)[0]!;
    if (extra < 248) {
      result += BASE62_CHARS[extra % 62];
    }
  }
  return result;
}

/** Generate a random CUID-like ID for non-Prisma use cases. */
export function generateId(): string {
  return crypto.randomBytes(10).toString('hex');
}
