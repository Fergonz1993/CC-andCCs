/**
 * JWT Authentication Middleware for MCP Server (PROD-005)
 *
 * Implements JWT-based authentication with:
 * - Token validation and verification
 * - Configurable secret via environment variable
 * - Token refresh endpoint support
 * - Role-based access control
 * - Optional auth bypass for local development
 *
 * Features:
 * - HS256 JWT signing and verification
 * - Token expiration handling
 * - Refresh token support
 * - Permission-based authorization
 * - Audit logging integration
 */

import * as crypto from "crypto";
import { v4 as uuidv4 } from "uuid";

// ============================================================================
// Types
// ============================================================================

export type AuthRole = "leader" | "worker" | "admin";

export type AuthPermission =
  | "read"
  | "write"
  | "admin"
  | "create_task"
  | "claim_task"
  | "complete_task"
  | "manage_agents"
  | "view_status"
  | "manage_webhooks"
  | "manage_subscriptions";

export interface JWTPayload {
  /** Subject - unique identifier for the token holder */
  sub: string;
  /** Issuer - who issued the token */
  iss: string;
  /** Audience - intended recipient */
  aud: string;
  /** Expiration time (Unix timestamp in seconds) */
  exp: number;
  /** Issued at time (Unix timestamp in seconds) */
  iat: number;
  /** JWT ID - unique identifier for this token */
  jti: string;
  /** User role */
  role: AuthRole;
  /** User permissions */
  permissions: AuthPermission[];
  /** Agent ID if applicable */
  agentId?: string;
  /** Custom claims */
  claims?: Record<string, unknown>;
}

export interface AuthConfig {
  /** JWT secret key (can also be set via JWT_SECRET env var) */
  secret: string;
  /** Token issuer name */
  issuer: string;
  /** Token audience */
  audience: string;
  /** Access token expiration in seconds (default: 3600 = 1 hour) */
  accessTokenExpirationSeconds: number;
  /** Refresh token expiration in seconds (default: 604800 = 7 days) */
  refreshTokenExpirationSeconds: number;
  /** Seconds before expiration when refresh is recommended */
  refreshThresholdSeconds: number;
  /** Whether authentication is enabled (default: true) */
  enabled: boolean;
  /** Whether to allow unauthenticated access in development mode */
  allowUnauthenticatedInDev: boolean;
  /** List of tools that don't require authentication */
  publicTools: string[];
  /** Callback for auth events */
  onAuthEvent?: (event: AuthEvent) => void;
}

export interface AuthRequest {
  /** JWT token from Authorization header */
  token?: string;
  /** API key as alternative auth method */
  apiKey?: string;
  /** Tool being accessed */
  toolName?: string;
  /** Agent ID making the request */
  agentId?: string;
  /** Client IP address */
  ip?: string;
}

export interface AuthResult {
  /** Whether authentication succeeded */
  authenticated: boolean;
  /** Decoded JWT payload if authenticated */
  payload?: JWTPayload;
  /** Error message if authentication failed */
  error?: string;
  /** Error code for programmatic handling */
  errorCode?: AuthErrorCode;
  /** Whether token should be refreshed soon */
  shouldRefresh?: boolean;
  /** New token if auto-refresh is enabled */
  newToken?: string;
}

export type AuthErrorCode =
  | "MISSING_TOKEN"
  | "INVALID_TOKEN"
  | "TOKEN_EXPIRED"
  | "INVALID_SIGNATURE"
  | "INVALID_ISSUER"
  | "INVALID_AUDIENCE"
  | "INSUFFICIENT_PERMISSIONS"
  | "AUTH_DISABLED"
  | "INVALID_ROLE";

export interface TokenPair {
  /** Access token for API requests */
  accessToken: string;
  /** Refresh token for obtaining new access tokens */
  refreshToken: string;
  /** Access token expiration timestamp (Unix ms) */
  expiresAt: number;
  /** Refresh token expiration timestamp (Unix ms) */
  refreshExpiresAt: number;
  /** Token type (always "Bearer") */
  tokenType: "Bearer";
}

export interface AuthEvent {
  type:
    | "login"
    | "logout"
    | "refresh"
    | "auth_failed"
    | "permission_denied"
    | "token_issued";
  subject?: string;
  role?: AuthRole;
  reason?: string;
  timestamp: string;
  ip?: string;
  toolName?: string;
}

// Refresh token storage
interface StoredRefreshToken {
  token: string;
  subject: string;
  role: AuthRole;
  permissions: AuthPermission[];
  agentId?: string;
  expiresAt: number;
  createdAt: number;
  usedAt?: number;
}

// ============================================================================
// Default Configuration
// ============================================================================

export const DEFAULT_AUTH_CONFIG: AuthConfig = {
  secret: process.env.JWT_SECRET || "",
  issuer: "claude-coordination-mcp",
  audience: "claude-agents",
  accessTokenExpirationSeconds: 3600, // 1 hour
  refreshTokenExpirationSeconds: 604800, // 7 days
  refreshThresholdSeconds: 300, // 5 minutes before expiration
  enabled: true,
  allowUnauthenticatedInDev: process.env.NODE_ENV === "development",
  publicTools: ["get_status", "get_master_plan", "health_check"],
};

// Default permissions per role
export const ROLE_PERMISSIONS: Record<AuthRole, AuthPermission[]> = {
  admin: [
    "read",
    "write",
    "admin",
    "create_task",
    "claim_task",
    "complete_task",
    "manage_agents",
    "view_status",
    "manage_webhooks",
    "manage_subscriptions",
  ],
  leader: [
    "read",
    "write",
    "create_task",
    "manage_agents",
    "view_status",
    "manage_webhooks",
    "manage_subscriptions",
  ],
  worker: ["read", "claim_task", "complete_task", "view_status"],
};

// Tool permission requirements
export const TOOL_PERMISSIONS: Record<string, AuthPermission[]> = {
  init_coordination: ["admin", "write"],
  create_task: ["create_task", "write"],
  create_tasks_batch: ["create_task", "write"],
  get_status: ["view_status", "read"],
  get_all_tasks: ["view_status", "read"],
  get_results: ["view_status", "read"],
  register_agent: ["manage_agents", "write"],
  claim_task: ["claim_task"],
  start_task: ["claim_task"],
  complete_task: ["complete_task"],
  fail_task: ["complete_task"],
  heartbeat: ["read"],
  add_discovery: ["write"],
  get_discoveries: ["read"],
  get_master_plan: ["read"],
  // Subscription tools
  subscribe: ["manage_subscriptions", "write"],
  unsubscribe: ["manage_subscriptions", "write"],
  get_notifications: ["read"],
  // Webhook tools
  register_webhook: ["manage_webhooks", "admin"],
  unregister_webhook: ["manage_webhooks", "admin"],
  // Admin tools
  get_audit_log: ["admin"],
  rotate_secrets: ["admin"],
};

// ============================================================================
// Auth Middleware Class
// ============================================================================

export class AuthMiddleware {
  private config: AuthConfig;
  private refreshTokens: Map<string, StoredRefreshToken> = new Map();
  private revokedTokens: Set<string> = new Set();
  private cleanupInterval: NodeJS.Timeout | null = null;
  private readonly CLEANUP_INTERVAL_MS = 300000; // 5 minutes

  constructor(config: Partial<AuthConfig> = {}) {
    this.config = { ...DEFAULT_AUTH_CONFIG, ...config };

    // Use environment variable if no secret provided
    if (!this.config.secret && process.env.JWT_SECRET) {
      this.config.secret = process.env.JWT_SECRET;
    }

    // Generate a random secret if none provided (for development)
    if (!this.config.secret) {
      this.config.secret = crypto.randomBytes(32).toString("hex");
      console.error(
        "[AuthMiddleware] Warning: Using auto-generated JWT secret. Set JWT_SECRET env var for production.",
      );
    }

    this.startCleanupInterval();
  }

  /**
   * Authenticate a request
   */
  authenticate(request: AuthRequest): AuthResult {
    // Check if auth is disabled
    if (!this.config.enabled) {
      return { authenticated: true };
    }

    // Check if this is a public tool
    if (
      request.toolName &&
      this.config.publicTools.includes(request.toolName)
    ) {
      return { authenticated: true };
    }

    // Check for development mode bypass
    if (this.config.allowUnauthenticatedInDev && !request.token) {
      return { authenticated: true };
    }

    // Require token
    if (!request.token) {
      this.emitAuthEvent({
        type: "auth_failed",
        reason: "Missing authentication token",
        timestamp: new Date().toISOString(),
        ip: request.ip,
        toolName: request.toolName,
      });

      return {
        authenticated: false,
        error: "Authentication required",
        errorCode: "MISSING_TOKEN",
      };
    }

    // Validate token
    try {
      const payload = this.verifyToken(request.token);

      // Check if token is revoked
      if (this.revokedTokens.has(payload.jti)) {
        return {
          authenticated: false,
          error: "Token has been revoked",
          errorCode: "INVALID_TOKEN",
        };
      }

      // Check if token should be refreshed soon
      const now = Math.floor(Date.now() / 1000);
      const shouldRefresh =
        payload.exp - now <= this.config.refreshThresholdSeconds;

      return {
        authenticated: true,
        payload,
        shouldRefresh,
      };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Invalid token";
      let errorCode: AuthErrorCode = "INVALID_TOKEN";

      if (errorMessage.includes("expired")) {
        errorCode = "TOKEN_EXPIRED";
      } else if (errorMessage.includes("signature")) {
        errorCode = "INVALID_SIGNATURE";
      } else if (errorMessage.includes("issuer")) {
        errorCode = "INVALID_ISSUER";
      } else if (errorMessage.includes("audience")) {
        errorCode = "INVALID_AUDIENCE";
      }

      this.emitAuthEvent({
        type: "auth_failed",
        reason: errorMessage,
        timestamp: new Date().toISOString(),
        ip: request.ip,
        toolName: request.toolName,
      });

      return {
        authenticated: false,
        error: errorMessage,
        errorCode,
      };
    }
  }

  /**
   * Check if a user has permission to access a tool
   */
  authorize(
    payload: JWTPayload | undefined,
    toolName: string,
  ): { authorized: boolean; error?: string } {
    // If no payload (auth disabled), allow all
    if (!payload) {
      if (!this.config.enabled) {
        return { authorized: true };
      }
      return { authorized: false, error: "Not authenticated" };
    }

    // Admin role has all permissions
    if (payload.role === "admin") {
      return { authorized: true };
    }

    // Check tool-specific permissions
    const requiredPermissions = TOOL_PERMISSIONS[toolName];
    if (!requiredPermissions) {
      // If tool has no specific requirements, allow read access
      return { authorized: payload.permissions.includes("read") };
    }

    // Check if user has any of the required permissions
    const hasPermission = requiredPermissions.some((perm) =>
      payload.permissions.includes(perm),
    );

    if (!hasPermission) {
      this.emitAuthEvent({
        type: "permission_denied",
        subject: payload.sub,
        role: payload.role,
        reason: `Missing required permission for ${toolName}`,
        timestamp: new Date().toISOString(),
        toolName,
      });

      return {
        authorized: false,
        error: `Insufficient permissions for ${toolName}. Required: ${requiredPermissions.join(" or ")}`,
      };
    }

    return { authorized: true };
  }

  /**
   * Generate a new token pair (access + refresh tokens)
   */
  generateTokenPair(
    subject: string,
    role: AuthRole,
    permissions?: AuthPermission[],
    agentId?: string,
    claims?: Record<string, unknown>,
  ): TokenPair {
    const effectivePermissions = permissions || ROLE_PERMISSIONS[role];
    const now = Math.floor(Date.now() / 1000);

    // Generate access token
    const accessPayload: JWTPayload = {
      sub: subject,
      iss: this.config.issuer,
      aud: this.config.audience,
      exp: now + this.config.accessTokenExpirationSeconds,
      iat: now,
      jti: uuidv4(),
      role,
      permissions: effectivePermissions,
      agentId,
      claims,
    };

    const accessToken = this.signToken(accessPayload);

    // Generate refresh token
    const refreshTokenId = uuidv4();
    const refreshToken = this.signToken({
      sub: subject,
      iss: this.config.issuer,
      aud: this.config.audience,
      exp: now + this.config.refreshTokenExpirationSeconds,
      iat: now,
      jti: refreshTokenId,
      role,
      permissions: effectivePermissions,
      agentId,
      claims: { type: "refresh" },
    });

    // Store refresh token
    this.refreshTokens.set(refreshTokenId, {
      token: refreshToken,
      subject,
      role,
      permissions: effectivePermissions,
      agentId,
      expiresAt: (now + this.config.refreshTokenExpirationSeconds) * 1000,
      createdAt: Date.now(),
    });

    this.emitAuthEvent({
      type: "token_issued",
      subject,
      role,
      timestamp: new Date().toISOString(),
    });

    return {
      accessToken,
      refreshToken,
      expiresAt: (now + this.config.accessTokenExpirationSeconds) * 1000,
      refreshExpiresAt:
        (now + this.config.refreshTokenExpirationSeconds) * 1000,
      tokenType: "Bearer",
    };
  }

  /**
   * Refresh an access token using a refresh token
   */
  refreshAccessToken(refreshToken: string): TokenPair | null {
    try {
      const payload = this.verifyToken(refreshToken);

      // Verify it's a refresh token
      if (payload.claims?.type !== "refresh") {
        return null;
      }

      // Check if refresh token is stored and valid
      const storedToken = this.refreshTokens.get(payload.jti);
      if (!storedToken || storedToken.expiresAt < Date.now()) {
        return null;
      }

      // Check if token is revoked
      if (this.revokedTokens.has(payload.jti)) {
        return null;
      }

      // Mark old refresh token as used
      storedToken.usedAt = Date.now();

      // Generate new token pair
      const newPair = this.generateTokenPair(
        storedToken.subject,
        storedToken.role,
        storedToken.permissions,
        storedToken.agentId,
      );

      // Revoke old refresh token (rotation)
      this.revokedTokens.add(payload.jti);
      this.refreshTokens.delete(payload.jti);

      this.emitAuthEvent({
        type: "refresh",
        subject: storedToken.subject,
        role: storedToken.role,
        timestamp: new Date().toISOString(),
      });

      return newPair;
    } catch {
      return null;
    }
  }

  /**
   * Revoke a token (logout)
   */
  revokeToken(token: string): boolean {
    try {
      const payload = this.verifyToken(token);
      this.revokedTokens.add(payload.jti);

      // If it's a refresh token, also remove from storage
      if (payload.claims?.type === "refresh") {
        this.refreshTokens.delete(payload.jti);
      }

      this.emitAuthEvent({
        type: "logout",
        subject: payload.sub,
        role: payload.role,
        timestamp: new Date().toISOString(),
      });

      return true;
    } catch {
      return false;
    }
  }

  /**
   * Revoke all tokens for a subject
   */
  revokeAllTokens(subject: string): number {
    let count = 0;

    // Revoke all refresh tokens for this subject
    for (const [id, token] of this.refreshTokens.entries()) {
      if (token.subject === subject) {
        this.revokedTokens.add(id);
        this.refreshTokens.delete(id);
        count++;
      }
    }

    return count;
  }

  /**
   * Validate a token without full authentication
   */
  validateToken(token: string): { valid: boolean; payload?: JWTPayload } {
    try {
      const payload = this.verifyToken(token);
      return { valid: true, payload };
    } catch {
      return { valid: false };
    }
  }

  /**
   * Check if a token should be refreshed
   */
  shouldRefreshToken(token: string): boolean {
    try {
      const payload = this.verifyToken(token);
      const now = Math.floor(Date.now() / 1000);
      return payload.exp - now <= this.config.refreshThresholdSeconds;
    } catch {
      return false;
    }
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<AuthConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration (without exposing secret)
   */
  getConfig(): Omit<AuthConfig, "secret"> & { secretConfigured: boolean } {
    const { secret: _secret, ...configWithoutSecret } = this.config;
    return {
      ...configWithoutSecret,
      secretConfigured: !!this.config.secret,
    };
  }

  /**
   * Check if a tool is public (no auth required)
   */
  isPublicTool(toolName: string): boolean {
    return this.config.publicTools.includes(toolName);
  }

  /**
   * Add a tool to the public list
   */
  addPublicTool(toolName: string): void {
    if (!this.config.publicTools.includes(toolName)) {
      this.config.publicTools.push(toolName);
    }
  }

  /**
   * Remove a tool from the public list
   */
  removePublicTool(toolName: string): void {
    const index = this.config.publicTools.indexOf(toolName);
    if (index !== -1) {
      this.config.publicTools.splice(index, 1);
    }
  }

  /**
   * Get statistics about auth state
   */
  getStats(): {
    activeRefreshTokens: number;
    revokedTokens: number;
    enabled: boolean;
    publicToolsCount: number;
  } {
    return {
      activeRefreshTokens: this.refreshTokens.size,
      revokedTokens: this.revokedTokens.size,
      enabled: this.config.enabled,
      publicToolsCount: this.config.publicTools.length,
    };
  }

  /**
   * Clean up resources
   */
  destroy(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
    this.refreshTokens.clear();
    this.revokedTokens.clear();
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private signToken(payload: JWTPayload): string {
    const header = { alg: "HS256", typ: "JWT" };
    const headerB64 = Buffer.from(JSON.stringify(header)).toString("base64url");
    const payloadB64 = Buffer.from(JSON.stringify(payload)).toString(
      "base64url",
    );

    const signature = crypto
      .createHmac("sha256", this.config.secret)
      .update(`${headerB64}.${payloadB64}`)
      .digest("base64url");

    return `${headerB64}.${payloadB64}.${signature}`;
  }

  private verifyToken(token: string): JWTPayload {
    const parts = token.split(".");
    if (parts.length !== 3) {
      throw new Error("Invalid token format");
    }

    const [headerB64, payloadB64, signature] = parts;

    // Verify signature
    const expectedSignature = crypto
      .createHmac("sha256", this.config.secret)
      .update(`${headerB64}.${payloadB64}`)
      .digest("base64url");

    if (
      !crypto.timingSafeEqual(
        Buffer.from(signature),
        Buffer.from(expectedSignature),
      )
    ) {
      throw new Error("Invalid token signature");
    }

    // Decode and validate payload
    const payload: JWTPayload = JSON.parse(
      Buffer.from(payloadB64, "base64url").toString(),
    );

    // Check expiration
    const now = Math.floor(Date.now() / 1000);
    if (payload.exp < now) {
      throw new Error("Token has expired");
    }

    // Validate issuer
    if (payload.iss !== this.config.issuer) {
      throw new Error("Invalid token issuer");
    }

    // Validate audience
    if (payload.aud !== this.config.audience) {
      throw new Error("Invalid token audience");
    }

    return payload;
  }

  private emitAuthEvent(event: AuthEvent): void {
    if (this.config.onAuthEvent) {
      try {
        this.config.onAuthEvent(event);
      } catch {
        // Ignore callback errors
      }
    }
  }

  private startCleanupInterval(): void {
    this.cleanupInterval = setInterval(() => {
      this.cleanup();
    }, this.CLEANUP_INTERVAL_MS);

    if (this.cleanupInterval.unref) {
      this.cleanupInterval.unref();
    }
  }

  private cleanup(): void {
    const now = Date.now();

    // Clean up expired refresh tokens
    for (const [id, token] of this.refreshTokens.entries()) {
      if (token.expiresAt < now) {
        this.refreshTokens.delete(id);
      }
    }

    // Limit revoked tokens set size (keep last 10000)
    if (this.revokedTokens.size > 10000) {
      const tokens = Array.from(this.revokedTokens);
      this.revokedTokens.clear();
      for (const token of tokens.slice(-10000)) {
        this.revokedTokens.add(token);
      }
    }
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create an auth middleware instance
 */
export function createAuthMiddleware(
  config: Partial<AuthConfig> = {},
): AuthMiddleware {
  return new AuthMiddleware(config);
}

// ============================================================================
// Default Instance
// ============================================================================

let defaultInstance: AuthMiddleware | null = null;

/**
 * Get the default auth middleware instance
 */
export function getAuthMiddleware(
  config?: Partial<AuthConfig>,
): AuthMiddleware {
  if (!defaultInstance) {
    defaultInstance = new AuthMiddleware(config);
  }
  return defaultInstance;
}

/**
 * Reset the default instance (useful for testing)
 */
export function resetAuthMiddleware(): void {
  if (defaultInstance) {
    defaultInstance.destroy();
    defaultInstance = null;
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Extract Bearer token from Authorization header
 */
export function extractBearerToken(
  authHeader: string | undefined,
): string | undefined {
  if (!authHeader) return undefined;
  const parts = authHeader.split(" ");
  if (parts.length === 2 && parts[0].toLowerCase() === "bearer") {
    return parts[1];
  }
  return undefined;
}

/**
 * Get default permissions for a role
 */
export function getPermissionsForRole(role: AuthRole): AuthPermission[] {
  return [...ROLE_PERMISSIONS[role]];
}

/**
 * Check if a permission is valid
 */
export function isValidPermission(
  permission: string,
): permission is AuthPermission {
  const validPermissions: AuthPermission[] = [
    "read",
    "write",
    "admin",
    "create_task",
    "claim_task",
    "complete_task",
    "manage_agents",
    "view_status",
    "manage_webhooks",
    "manage_subscriptions",
  ];
  return validPermissions.includes(permission as AuthPermission);
}

/**
 * Check if a role is valid
 */
export function isValidRole(role: string): role is AuthRole {
  return ["leader", "worker", "admin"].includes(role);
}
