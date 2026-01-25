/**
 * Middleware exports for MCP Server
 *
 * This module provides security middleware for the MCP server:
 * - Rate limiting (PROD-004)
 * - JWT authentication (PROD-005)
 */

// Rate Limiting Middleware (PROD-004)
export {
  RateLimitMiddleware,
  createRateLimitMiddleware,
  getRateLimitMiddleware,
  resetRateLimitMiddleware,
  DEFAULT_RATE_LIMIT_CONFIG,
  type RateLimitConfig,
  type RateLimitRequest,
  type RateLimitResult,
  type RateLimitInfo,
  type RateLimitHeaders,
} from "./rateLimit.js";

// JWT Authentication Middleware (PROD-005)
export {
  AuthMiddleware,
  createAuthMiddleware,
  getAuthMiddleware,
  resetAuthMiddleware,
  extractBearerToken,
  getPermissionsForRole,
  isValidPermission,
  isValidRole,
  DEFAULT_AUTH_CONFIG,
  ROLE_PERMISSIONS,
  TOOL_PERMISSIONS,
  type AuthConfig,
  type AuthRequest,
  type AuthResult,
  type AuthRole,
  type AuthPermission,
  type AuthErrorCode,
  type AuthEvent,
  type JWTPayload,
  type TokenPair,
} from "./auth.js";
