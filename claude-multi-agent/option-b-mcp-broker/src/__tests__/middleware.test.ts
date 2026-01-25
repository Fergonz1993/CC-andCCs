/**
 * Tests for Rate Limiting and JWT Authentication Middleware (PROD-004, PROD-005)
 *
 * Run with: npm test -- --testPathPattern=middleware
 */

import { describe, it, expect, beforeEach, afterEach } from "@jest/globals";
import {
  RateLimitMiddleware,
  createRateLimitMiddleware,
  resetRateLimitMiddleware,
  type RateLimitConfig,
  type RateLimitRequest,
} from "../middleware/rateLimit.js";
import {
  AuthMiddleware,
  createAuthMiddleware,
  resetAuthMiddleware,
  extractBearerToken,
  getPermissionsForRole,
  isValidPermission,
  isValidRole,
  ROLE_PERMISSIONS,
  type AuthConfig,
  type AuthRole,
} from "../middleware/auth.js";

// ============================================================================
// Rate Limit Middleware Tests (PROD-004)
// ============================================================================

describe("RateLimitMiddleware (PROD-004)", () => {
  let middleware: RateLimitMiddleware;

  beforeEach(() => {
    resetRateLimitMiddleware();
    middleware = createRateLimitMiddleware({
      maxRequests: 10,
      windowMs: 1000,
      burstLimit: 20, // Set burst limit higher than maxRequests for basic tests
      burstWindowMs: 100,
    });
  });

  afterEach(() => {
    middleware.destroy();
  });

  describe("Basic Rate Limiting", () => {
    it("should allow requests under the limit", () => {
      const request: RateLimitRequest = { clientId: "client-1" };

      for (let i = 0; i < 5; i++) {
        const result = middleware.checkLimit(request);
        expect(result.allowed).toBe(true);
        expect(result.remaining).toBe(10 - (i + 1));
      }
    });

    it("should deny requests over the limit", () => {
      const request: RateLimitRequest = { clientId: "client-2" };

      // Use up all requests
      for (let i = 0; i < 10; i++) {
        middleware.checkLimit(request);
      }

      // Next request should be denied
      const result = middleware.checkLimit(request);
      expect(result.allowed).toBe(false);
      expect(result.remaining).toBe(0);
      expect(result.retryAfter).toBeDefined();
    });

    it("should track requests per client separately", () => {
      const client1: RateLimitRequest = { clientId: "client-a" };
      const client2: RateLimitRequest = { clientId: "client-b" };

      // Use up client1's requests
      for (let i = 0; i < 10; i++) {
        middleware.checkLimit(client1);
      }

      // Client2 should still be allowed
      const result = middleware.checkLimit(client2);
      expect(result.allowed).toBe(true);
    });

    it("should use agentId as client key when clientId not provided", () => {
      const request: RateLimitRequest = { agentId: "agent-1" };
      const result = middleware.checkLimit(request);

      expect(result.allowed).toBe(true);
      expect(result.info.clientId).toBe("agent:agent-1");
    });

    it("should use IP as client key when only IP provided", () => {
      const request: RateLimitRequest = { ip: "192.168.1.1" };
      const result = middleware.checkLimit(request);

      expect(result.allowed).toBe(true);
      expect(result.info.clientId).toBe("ip:192.168.1.1");
    });
  });

  describe("Burst Limiting", () => {
    it("should enforce burst limits", () => {
      // Create a middleware with low burst limit for this specific test
      const burstMiddleware = createRateLimitMiddleware({
        maxRequests: 100,
        windowMs: 1000,
        burstLimit: 3,
        burstWindowMs: 100,
      });

      const request: RateLimitRequest = { clientId: "burst-client" };

      // Send 3 requests quickly (at burst limit)
      for (let i = 0; i < 3; i++) {
        const result = burstMiddleware.checkLimit(request);
        expect(result.allowed).toBe(true);
      }

      // 4th request should be denied due to burst limit
      const result = burstMiddleware.checkLimit(request);
      expect(result.allowed).toBe(false);
      expect(result.info.burstViolation).toBe(true);

      burstMiddleware.destroy();
    });
  });

  describe("Rate Limit Headers", () => {
    it("should generate correct headers for allowed request", () => {
      const request: RateLimitRequest = { clientId: "header-client" };
      const result = middleware.checkLimit(request);
      const headers = middleware.getHeaders(result);

      expect(headers["X-RateLimit-Limit"]).toBe("10");
      expect(headers["X-RateLimit-Remaining"]).toBeDefined();
      expect(headers["X-RateLimit-Reset"]).toBeDefined();
      expect(headers["X-RateLimit-Policy"]).toContain("10;w=");
      expect(headers["Retry-After"]).toBeUndefined();
    });

    it("should include Retry-After for denied request", () => {
      const request: RateLimitRequest = { clientId: "denied-client" };

      // Exhaust limit
      for (let i = 0; i < 10; i++) {
        middleware.checkLimit(request);
      }

      const result = middleware.checkLimit(request);
      const headers = middleware.getHeaders(result);

      expect(headers["Retry-After"]).toBeDefined();
    });
  });

  describe("Token Consumption", () => {
    it("should consume multiple tokens at once", () => {
      const request: RateLimitRequest = { clientId: "batch-client" };

      const result = middleware.consumeTokens(request, 5);
      expect(result.allowed).toBe(true);
      expect(result.remaining).toBe(5);
    });

    it("should deny if not enough tokens available", () => {
      const request: RateLimitRequest = { clientId: "batch-deny" };

      // Use some tokens
      middleware.consumeTokens(request, 8);

      // Try to consume more than available
      const result = middleware.consumeTokens(request, 5);
      expect(result.allowed).toBe(false);
    });
  });

  describe("Skip List", () => {
    it("should skip rate limiting for clients in skip list", () => {
      middleware.addToSkipList("vip-client");
      const request: RateLimitRequest = { clientId: "vip-client" };

      // Should always be allowed
      for (let i = 0; i < 20; i++) {
        const result = middleware.checkLimit(request);
        expect(result.allowed).toBe(true);
      }
    });

    it("should remove client from skip list", () => {
      middleware.addToSkipList("temp-vip");
      middleware.removeFromSkipList("temp-vip");

      const request: RateLimitRequest = { clientId: "temp-vip" };

      // Exhaust limit
      for (let i = 0; i < 10; i++) {
        middleware.checkLimit(request);
      }

      const result = middleware.checkLimit(request);
      expect(result.allowed).toBe(false);
    });
  });

  describe("Client Status", () => {
    it("should return correct client status", () => {
      const request: RateLimitRequest = { clientId: "status-client" };

      // Make some requests
      for (let i = 0; i < 3; i++) {
        middleware.checkLimit(request);
      }

      const status = middleware.getClientStatus("status-client");
      expect(status).not.toBeNull();
      expect(status!.requests).toBe(3);
      expect(status!.remaining).toBe(7);
      expect(status!.limit).toBe(10);
    });

    it("should return default status for unknown client", () => {
      const status = middleware.getClientStatus("unknown-client");
      expect(status).not.toBeNull();
      expect(status!.requests).toBe(0);
      expect(status!.remaining).toBe(10);
    });
  });

  describe("Reset Operations", () => {
    it("should reset rate limit for specific client", () => {
      const request: RateLimitRequest = { clientId: "reset-client" };

      // Exhaust limit
      for (let i = 0; i < 10; i++) {
        middleware.checkLimit(request);
      }

      // Reset
      middleware.reset("reset-client");

      // Should be allowed again
      const result = middleware.checkLimit(request);
      expect(result.allowed).toBe(true);
    });

    it("should reset all rate limits", () => {
      // Create multiple clients
      for (let i = 0; i < 5; i++) {
        const request: RateLimitRequest = { clientId: `client-${i}` };
        middleware.checkLimit(request);
      }

      middleware.resetAll();

      const stats = middleware.getStats();
      expect(stats.totalClients).toBe(0);
    });
  });

  describe("Configuration", () => {
    it("should update configuration", () => {
      middleware.updateConfig({ maxRequests: 50 });
      const config = middleware.getConfig();
      expect(config.maxRequests).toBe(50);
    });

    it("should disable rate limiting when enabled is false", () => {
      middleware.updateConfig({ enabled: false });

      const request: RateLimitRequest = { clientId: "disabled-client" };

      // Should always be allowed
      for (let i = 0; i < 100; i++) {
        const result = middleware.checkLimit(request);
        expect(result.allowed).toBe(true);
      }
    });
  });

  describe("Statistics", () => {
    it("should return correct statistics", () => {
      // Make requests from multiple clients
      for (let i = 0; i < 3; i++) {
        const request: RateLimitRequest = { clientId: `stat-client-${i}` };
        middleware.checkLimit(request);
        middleware.checkLimit(request);
      }

      const stats = middleware.getStats();
      expect(stats.totalClients).toBe(3);
      expect(stats.activeClients).toBe(3);
      expect(stats.totalRequests).toBe(6);
    });
  });

  describe("Custom Key Generator", () => {
    it("should use custom key generator when provided", () => {
      const customMiddleware = createRateLimitMiddleware({
        maxRequests: 10,
        windowMs: 1000,
        keyGenerator: (req) => `custom:${req.toolName || "unknown"}`,
      });

      const request: RateLimitRequest = { toolName: "create_task" };
      const result = customMiddleware.checkLimit(request);

      expect(result.info.clientId).toBe("custom:create_task");
      customMiddleware.destroy();
    });
  });

  describe("Rate Limit Exceeded Callback", () => {
    it("should call callback when limit exceeded", () => {
      let callbackCalled = false;
      let callbackClientId = "";

      const callbackMiddleware = createRateLimitMiddleware({
        maxRequests: 2,
        windowMs: 1000,
        burstLimit: 10,
        onRateLimitExceeded: (clientId) => {
          callbackCalled = true;
          callbackClientId = clientId;
        },
      });

      const request: RateLimitRequest = { clientId: "callback-client" };

      // Exhaust limit
      callbackMiddleware.checkLimit(request);
      callbackMiddleware.checkLimit(request);
      callbackMiddleware.checkLimit(request);

      expect(callbackCalled).toBe(true);
      expect(callbackClientId).toBe("callback-client");
      callbackMiddleware.destroy();
    });
  });
});

// ============================================================================
// Auth Middleware Tests (PROD-005)
// ============================================================================

describe("AuthMiddleware (PROD-005)", () => {
  let middleware: AuthMiddleware;
  const testSecret = "test-secret-key-for-jwt-signing-32chars";

  beforeEach(() => {
    resetAuthMiddleware();
    middleware = createAuthMiddleware({
      secret: testSecret,
      enabled: true,
      allowUnauthenticatedInDev: false,
    });
  });

  afterEach(() => {
    middleware.destroy();
  });

  describe("Token Generation", () => {
    it("should generate valid token pair", () => {
      const tokenPair = middleware.generateTokenPair("user-1", "worker");

      expect(tokenPair.accessToken).toBeDefined();
      expect(tokenPair.refreshToken).toBeDefined();
      expect(tokenPair.tokenType).toBe("Bearer");
      expect(tokenPair.expiresAt).toBeGreaterThan(Date.now());
    });

    it("should generate tokens with correct role permissions", () => {
      const tokenPair = middleware.generateTokenPair("admin-user", "admin");
      const validation = middleware.validateToken(tokenPair.accessToken);

      expect(validation.valid).toBe(true);
      expect(validation.payload?.role).toBe("admin");
      expect(validation.payload?.permissions).toContain("admin");
    });

    it("should include custom permissions when provided", () => {
      const tokenPair = middleware.generateTokenPair("custom-user", "worker", [
        "read",
        "write",
        "create_task",
      ]);
      const validation = middleware.validateToken(tokenPair.accessToken);

      expect(validation.payload?.permissions).toContain("create_task");
    });

    it("should include agent ID in token", () => {
      const tokenPair = middleware.generateTokenPair(
        "agent-user",
        "worker",
        undefined,
        "agent-123",
      );
      const validation = middleware.validateToken(tokenPair.accessToken);

      expect(validation.payload?.agentId).toBe("agent-123");
    });
  });

  describe("Token Validation", () => {
    it("should validate correct token", () => {
      const tokenPair = middleware.generateTokenPair("valid-user", "worker");
      const result = middleware.authenticate({ token: tokenPair.accessToken });

      expect(result.authenticated).toBe(true);
      expect(result.payload?.sub).toBe("valid-user");
    });

    it("should reject invalid token", () => {
      const result = middleware.authenticate({ token: "invalid.token.here" });

      expect(result.authenticated).toBe(false);
      expect(result.errorCode).toBe("INVALID_TOKEN");
    });

    it("should reject tampered token", () => {
      const tokenPair = middleware.generateTokenPair("tampered-user", "worker");
      const parts = tokenPair.accessToken.split(".");
      parts[1] = Buffer.from('{"sub":"hacker"}').toString("base64url");
      const tamperedToken = parts.join(".");

      const result = middleware.authenticate({ token: tamperedToken });

      expect(result.authenticated).toBe(false);
      expect(result.errorCode).toBe("INVALID_SIGNATURE");
    });

    it("should reject expired token", async () => {
      // Create middleware with very short expiration
      const shortLivedMiddleware = createAuthMiddleware({
        secret: testSecret,
        accessTokenExpirationSeconds: -1, // Already expired (in the past)
      });

      const tokenPair = shortLivedMiddleware.generateTokenPair(
        "expired-user",
        "worker",
      );

      // No wait needed - token is already expired

      const result = shortLivedMiddleware.authenticate({
        token: tokenPair.accessToken,
      });

      expect(result.authenticated).toBe(false);
      expect(result.errorCode).toBe("TOKEN_EXPIRED");

      shortLivedMiddleware.destroy();
    }, 5000); // Increase test timeout
  });

  describe("Authentication Flow", () => {
    it("should require token when auth is enabled", () => {
      const result = middleware.authenticate({});

      expect(result.authenticated).toBe(false);
      expect(result.errorCode).toBe("MISSING_TOKEN");
    });

    it("should allow requests when auth is disabled", () => {
      middleware.updateConfig({ enabled: false });
      const result = middleware.authenticate({});

      expect(result.authenticated).toBe(true);
    });

    it("should allow public tools without token", () => {
      const result = middleware.authenticate({ toolName: "get_status" });

      expect(result.authenticated).toBe(true);
    });

    it("should indicate when token should be refreshed", () => {
      const shortRefreshMiddleware = createAuthMiddleware({
        secret: testSecret,
        accessTokenExpirationSeconds: 10,
        refreshThresholdSeconds: 15, // Always recommend refresh
      });

      const tokenPair = shortRefreshMiddleware.generateTokenPair(
        "refresh-user",
        "worker",
      );
      const result = shortRefreshMiddleware.authenticate({
        token: tokenPair.accessToken,
      });

      expect(result.shouldRefresh).toBe(true);

      shortRefreshMiddleware.destroy();
    });
  });

  describe("Authorization", () => {
    it("should authorize admin for all tools", () => {
      const tokenPair = middleware.generateTokenPair("admin-user", "admin");
      const validation = middleware.validateToken(tokenPair.accessToken);

      const result = middleware.authorize(
        validation.payload,
        "init_coordination",
      );

      expect(result.authorized).toBe(true);
    });

    it("should authorize worker for claim_task", () => {
      const tokenPair = middleware.generateTokenPair("worker-user", "worker");
      const validation = middleware.validateToken(tokenPair.accessToken);

      const result = middleware.authorize(validation.payload, "claim_task");

      expect(result.authorized).toBe(true);
    });

    it("should deny worker for create_task", () => {
      const tokenPair = middleware.generateTokenPair("worker-user", "worker");
      const validation = middleware.validateToken(tokenPair.accessToken);

      const result = middleware.authorize(validation.payload, "create_task");

      expect(result.authorized).toBe(false);
      expect(result.error).toContain("Insufficient permissions");
    });

    it("should authorize leader for create_task", () => {
      const tokenPair = middleware.generateTokenPair("leader-user", "leader");
      const validation = middleware.validateToken(tokenPair.accessToken);

      const result = middleware.authorize(validation.payload, "create_task");

      expect(result.authorized).toBe(true);
    });
  });

  describe("Token Refresh", () => {
    it("should refresh access token with valid refresh token", () => {
      const tokenPair = middleware.generateTokenPair("refresh-user", "worker");
      const newPair = middleware.refreshAccessToken(tokenPair.refreshToken);

      expect(newPair).not.toBeNull();
      expect(newPair!.accessToken).not.toBe(tokenPair.accessToken);
    });

    it("should not refresh with access token", () => {
      const tokenPair = middleware.generateTokenPair("access-user", "worker");
      const newPair = middleware.refreshAccessToken(tokenPair.accessToken);

      expect(newPair).toBeNull();
    });

    it("should revoke old refresh token after use", () => {
      const tokenPair = middleware.generateTokenPair("rotate-user", "worker");

      // First refresh should work
      const newPair = middleware.refreshAccessToken(tokenPair.refreshToken);
      expect(newPair).not.toBeNull();

      // Second refresh with old token should fail
      const secondRefresh = middleware.refreshAccessToken(
        tokenPair.refreshToken,
      );
      expect(secondRefresh).toBeNull();
    });
  });

  describe("Token Revocation", () => {
    it("should revoke token", () => {
      const tokenPair = middleware.generateTokenPair("revoke-user", "worker");

      const revoked = middleware.revokeToken(tokenPair.accessToken);
      expect(revoked).toBe(true);

      const result = middleware.authenticate({ token: tokenPair.accessToken });
      expect(result.authenticated).toBe(false);
    });

    it("should revoke all tokens for a subject", () => {
      const tokenPair1 = middleware.generateTokenPair(
        "multi-token-user",
        "worker",
      );
      const tokenPair2 = middleware.generateTokenPair(
        "multi-token-user",
        "worker",
      );

      const count = middleware.revokeAllTokens("multi-token-user");
      expect(count).toBe(2);
    });
  });

  describe("Public Tools", () => {
    it("should identify public tools", () => {
      expect(middleware.isPublicTool("get_status")).toBe(true);
      expect(middleware.isPublicTool("create_task")).toBe(false);
    });

    it("should add tool to public list", () => {
      middleware.addPublicTool("custom_public_tool");
      expect(middleware.isPublicTool("custom_public_tool")).toBe(true);
    });

    it("should remove tool from public list", () => {
      middleware.removePublicTool("get_status");
      expect(middleware.isPublicTool("get_status")).toBe(false);
    });
  });

  describe("Configuration", () => {
    it("should return config without exposing secret", () => {
      const config = middleware.getConfig();

      expect(config.secretConfigured).toBe(true);
      expect((config as Record<string, unknown>).secret).toBeUndefined();
    });

    it("should update configuration", () => {
      middleware.updateConfig({ accessTokenExpirationSeconds: 7200 });
      const config = middleware.getConfig();

      expect(config.accessTokenExpirationSeconds).toBe(7200);
    });
  });

  describe("Statistics", () => {
    it("should return correct statistics", () => {
      // Generate some tokens
      middleware.generateTokenPair("stat-user-1", "worker");
      middleware.generateTokenPair("stat-user-2", "leader");

      const stats = middleware.getStats();

      expect(stats.activeRefreshTokens).toBe(2);
      expect(stats.enabled).toBe(true);
    });
  });

  describe("Auth Events", () => {
    it("should emit auth events", () => {
      let eventReceived = false;
      let eventType = "";

      const eventMiddleware = createAuthMiddleware({
        secret: testSecret,
        onAuthEvent: (event) => {
          eventReceived = true;
          eventType = event.type;
        },
      });

      eventMiddleware.generateTokenPair("event-user", "worker");

      expect(eventReceived).toBe(true);
      expect(eventType).toBe("token_issued");

      eventMiddleware.destroy();
    });
  });
});

// ============================================================================
// Utility Function Tests
// ============================================================================

describe("Auth Utility Functions", () => {
  describe("extractBearerToken", () => {
    it("should extract token from valid Bearer header", () => {
      const token = extractBearerToken("Bearer abc123");
      expect(token).toBe("abc123");
    });

    it("should be case-insensitive for Bearer", () => {
      const token = extractBearerToken("bearer abc123");
      expect(token).toBe("abc123");
    });

    it("should return undefined for invalid header", () => {
      expect(extractBearerToken("Basic abc123")).toBeUndefined();
      expect(extractBearerToken("abc123")).toBeUndefined();
      expect(extractBearerToken("")).toBeUndefined();
      expect(extractBearerToken(undefined)).toBeUndefined();
    });
  });

  describe("getPermissionsForRole", () => {
    it("should return correct permissions for admin", () => {
      const perms = getPermissionsForRole("admin");
      expect(perms).toContain("admin");
      expect(perms).toContain("create_task");
      expect(perms).toContain("claim_task");
    });

    it("should return correct permissions for leader", () => {
      const perms = getPermissionsForRole("leader");
      expect(perms).toContain("create_task");
      expect(perms).not.toContain("admin");
    });

    it("should return correct permissions for worker", () => {
      const perms = getPermissionsForRole("worker");
      expect(perms).toContain("claim_task");
      expect(perms).toContain("complete_task");
      expect(perms).not.toContain("create_task");
    });
  });

  describe("isValidPermission", () => {
    it("should return true for valid permissions", () => {
      expect(isValidPermission("read")).toBe(true);
      expect(isValidPermission("write")).toBe(true);
      expect(isValidPermission("admin")).toBe(true);
      expect(isValidPermission("create_task")).toBe(true);
    });

    it("should return false for invalid permissions", () => {
      expect(isValidPermission("invalid")).toBe(false);
      expect(isValidPermission("")).toBe(false);
    });
  });

  describe("isValidRole", () => {
    it("should return true for valid roles", () => {
      expect(isValidRole("admin")).toBe(true);
      expect(isValidRole("leader")).toBe(true);
      expect(isValidRole("worker")).toBe(true);
    });

    it("should return false for invalid roles", () => {
      expect(isValidRole("superuser")).toBe(false);
      expect(isValidRole("")).toBe(false);
    });
  });

  describe("ROLE_PERMISSIONS constant", () => {
    it("should have all required roles", () => {
      expect(ROLE_PERMISSIONS.admin).toBeDefined();
      expect(ROLE_PERMISSIONS.leader).toBeDefined();
      expect(ROLE_PERMISSIONS.worker).toBeDefined();
    });

    it("should have admin with most permissions", () => {
      expect(ROLE_PERMISSIONS.admin.length).toBeGreaterThan(
        ROLE_PERMISSIONS.leader.length,
      );
      expect(ROLE_PERMISSIONS.admin.length).toBeGreaterThan(
        ROLE_PERMISSIONS.worker.length,
      );
    });
  });
});

// ============================================================================
// Integration Tests
// ============================================================================

describe("Middleware Integration", () => {
  let rateLimiter: RateLimitMiddleware;
  let auth: AuthMiddleware;

  beforeEach(() => {
    rateLimiter = createRateLimitMiddleware({
      maxRequests: 100,
      windowMs: 60000,
    });
    auth = createAuthMiddleware({
      secret: "integration-test-secret-key-32ch",
      enabled: true,
    });
  });

  afterEach(() => {
    rateLimiter.destroy();
    auth.destroy();
  });

  it("should work together for authenticated rate-limited requests", () => {
    // Generate token
    const tokenPair = auth.generateTokenPair("integration-user", "worker");

    // Authenticate
    const authResult = auth.authenticate({ token: tokenPair.accessToken });
    expect(authResult.authenticated).toBe(true);

    // Rate limit check using subject as client ID
    const rateLimitResult = rateLimiter.checkLimit({
      clientId: authResult.payload?.sub,
    });
    expect(rateLimitResult.allowed).toBe(true);

    // Authorize for a specific tool
    const authzResult = auth.authorize(authResult.payload, "claim_task");
    expect(authzResult.authorized).toBe(true);
  });

  it("should deny rate-limited authenticated user", () => {
    const tokenPair = auth.generateTokenPair("limited-user", "worker");

    // Authenticate
    const authResult = auth.authenticate({ token: tokenPair.accessToken });
    expect(authResult.authenticated).toBe(true);

    // Exhaust rate limit
    for (let i = 0; i < 100; i++) {
      rateLimiter.checkLimit({ clientId: authResult.payload?.sub });
    }

    // Should be rate limited
    const rateLimitResult = rateLimiter.checkLimit({
      clientId: authResult.payload?.sub,
    });
    expect(rateLimitResult.allowed).toBe(false);
  });
});
