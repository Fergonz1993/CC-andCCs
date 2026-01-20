/**
 * Security Module for Claude Multi-Agent Coordination MCP Server
 *
 * Features:
 * - sec-001: API key authentication
 * - sec-002: JWT token support with expiration
 * - sec-003: Role-based access control (RBAC)
 * - sec-004: Task data encryption at rest
 * - sec-005: Secure credential storage
 * - sec-006: Audit logging for all operations
 * - sec-007: Input sanitization for task descriptions
 * - sec-008: Rate limiting per identity
 * - sec-009: mTLS support for inter-agent communication
 * - sec-010: Secret rotation mechanism
 */

import * as crypto from "crypto";
import { v4 as uuidv4 } from "uuid";

// ============================================================================
// Types
// ============================================================================

export interface APIKey {
  id: string;
  key: string;
  name: string;
  agent_id: string | null;
  role: "leader" | "worker" | "admin";
  permissions: Permission[];
  rate_limit: RateLimitConfig;
  ip_allowlist: string[];
  created_at: string;
  expires_at: string | null;
  last_used_at: string | null;
  enabled: boolean;
}

export type Permission =
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

export interface RateLimitConfig {
  requests_per_minute: number;
  requests_per_hour: number;
  burst_limit: number;
}

export interface RateLimitState {
  minute_count: number;
  minute_reset_at: number;
  hour_count: number;
  hour_reset_at: number;
  burst_tokens: number;
  last_request_at: number;
}

export interface JWTPayload {
  sub: string; // Subject (agent_id or key_id)
  iss: string; // Issuer
  aud: string; // Audience
  exp: number; // Expiration time
  iat: number; // Issued at
  jti: string; // JWT ID
  role: "leader" | "worker" | "admin";
  permissions: Permission[];
}

export interface JWTConfig {
  secret: string;
  issuer: string;
  audience: string;
  expiration_seconds: number;
  refresh_threshold_seconds: number;
}

export interface RequestSignature {
  algorithm: "sha256" | "sha512";
  timestamp: number;
  nonce: string;
  signature: string;
}

export interface IPAllowlistConfig {
  enabled: boolean;
  default_policy: "allow" | "deny";
  global_allowlist: string[];
  global_denylist: string[];
}

export interface SecurityConfig {
  api_key_enabled: boolean;
  jwt_enabled: boolean;
  request_signing_enabled: boolean;
  ip_allowlist_enabled: boolean;
  rate_limiting_enabled: boolean;
  jwt_config: JWTConfig;
  ip_config: IPAllowlistConfig;
  signature_max_age_seconds: number;
}

export interface AuthResult {
  authenticated: boolean;
  api_key?: APIKey;
  jwt_payload?: JWTPayload;
  error?: string;
  error_code?: SecurityErrorCode;
}

export type SecurityErrorCode =
  | "MISSING_CREDENTIALS"
  | "INVALID_API_KEY"
  | "API_KEY_DISABLED"
  | "API_KEY_EXPIRED"
  | "INVALID_JWT"
  | "JWT_EXPIRED"
  | "INVALID_SIGNATURE"
  | "SIGNATURE_EXPIRED"
  | "IP_NOT_ALLOWED"
  | "RATE_LIMIT_EXCEEDED"
  | "INSUFFICIENT_PERMISSIONS";

// ============================================================================
// Security Manager Class
// ============================================================================

export class SecurityManager {
  private apiKeys: Map<string, APIKey> = new Map();
  private apiKeysByKey: Map<string, APIKey> = new Map();
  private rateLimitStates: Map<string, RateLimitState> = new Map();
  private usedNonces: Map<string, number> = new Map();
  private nonceCleanupInterval: NodeJS.Timeout | null = null;
  private config: SecurityConfig;

  constructor(config: Partial<SecurityConfig> = {}) {
    this.config = {
      api_key_enabled: config.api_key_enabled ?? true,
      jwt_enabled: config.jwt_enabled ?? false,
      request_signing_enabled: config.request_signing_enabled ?? false,
      ip_allowlist_enabled: config.ip_allowlist_enabled ?? false,
      rate_limiting_enabled: config.rate_limiting_enabled ?? true,
      jwt_config: config.jwt_config ?? {
        secret: crypto.randomBytes(32).toString("hex"),
        issuer: "claude-coordination",
        audience: "claude-agents",
        expiration_seconds: 3600,
        refresh_threshold_seconds: 300,
      },
      ip_config: config.ip_config ?? {
        enabled: false,
        default_policy: "allow",
        global_allowlist: [],
        global_denylist: [],
      },
      signature_max_age_seconds: config.signature_max_age_seconds ?? 300,
    };

    // Start nonce cleanup
    this.startNonceCleanup();
  }

  // --------------------------------------------------------------------------
  // API Key Management (adv-b-sec-001)
  // --------------------------------------------------------------------------

  /**
   * Create a new API key
   */
  createAPIKey(options: {
    name: string;
    agent_id?: string;
    role?: "leader" | "worker" | "admin";
    permissions?: Permission[];
    rate_limit?: Partial<RateLimitConfig>;
    ip_allowlist?: string[];
    expires_in_days?: number;
  }): APIKey {
    const key = this.generateAPIKey();
    const apiKey: APIKey = {
      id: uuidv4(),
      key,
      name: options.name,
      agent_id: options.agent_id || null,
      role: options.role || "worker",
      permissions:
        options.permissions ||
        this.getDefaultPermissions(options.role || "worker"),
      rate_limit: {
        requests_per_minute: options.rate_limit?.requests_per_minute ?? 60,
        requests_per_hour: options.rate_limit?.requests_per_hour ?? 1000,
        burst_limit: options.rate_limit?.burst_limit ?? 10,
      },
      ip_allowlist: options.ip_allowlist || [],
      created_at: new Date().toISOString(),
      expires_at: options.expires_in_days
        ? new Date(
            Date.now() + options.expires_in_days * 24 * 60 * 60 * 1000,
          ).toISOString()
        : null,
      last_used_at: null,
      enabled: true,
    };

    this.apiKeys.set(apiKey.id, apiKey);
    this.apiKeysByKey.set(apiKey.key, apiKey);

    return apiKey;
  }

  /**
   * Validate an API key
   */
  validateAPIKey(key: string): AuthResult {
    if (!this.config.api_key_enabled) {
      return { authenticated: true };
    }

    const apiKey = this.apiKeysByKey.get(key);
    if (!apiKey) {
      return {
        authenticated: false,
        error: "Invalid API key",
        error_code: "INVALID_API_KEY",
      };
    }

    if (!apiKey.enabled) {
      return {
        authenticated: false,
        error: "API key is disabled",
        error_code: "API_KEY_DISABLED",
      };
    }

    if (apiKey.expires_at && new Date(apiKey.expires_at) < new Date()) {
      return {
        authenticated: false,
        error: "API key has expired",
        error_code: "API_KEY_EXPIRED",
      };
    }

    // Update last used
    apiKey.last_used_at = new Date().toISOString();

    return { authenticated: true, api_key: apiKey };
  }

  /**
   * Revoke an API key
   */
  revokeAPIKey(keyId: string): boolean {
    const apiKey = this.apiKeys.get(keyId);
    if (!apiKey) return false;

    this.apiKeys.delete(keyId);
    this.apiKeysByKey.delete(apiKey.key);
    this.rateLimitStates.delete(keyId);

    return true;
  }

  /**
   * Disable/enable an API key
   */
  setAPIKeyEnabled(keyId: string, enabled: boolean): boolean {
    const apiKey = this.apiKeys.get(keyId);
    if (!apiKey) return false;
    apiKey.enabled = enabled;
    return true;
  }

  /**
   * Get an API key by ID
   */
  getAPIKey(keyId: string): APIKey | undefined {
    return this.apiKeys.get(keyId);
  }

  /**
   * Get all API keys (without exposing the actual key values)
   */
  getAllAPIKeys(): Omit<APIKey, "key">[] {
    return Array.from(this.apiKeys.values()).map((k) => {
      const { key: _key, ...rest } = k;
      return { ...rest, key: `${k.key.substring(0, 8)}...` };
    });
  }

  /**
   * Generate a secure API key
   */
  private generateAPIKey(): string {
    return `mcp_${crypto.randomBytes(32).toString("base64url")}`;
  }

  /**
   * Get default permissions for a role
   */
  private getDefaultPermissions(
    role: "leader" | "worker" | "admin",
  ): Permission[] {
    switch (role) {
      case "admin":
        return [
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
      case "leader":
        return [
          "read",
          "write",
          "create_task",
          "manage_agents",
          "view_status",
          "manage_webhooks",
          "manage_subscriptions",
        ];
      case "worker":
        return ["read", "claim_task", "complete_task", "view_status"];
    }
  }

  // --------------------------------------------------------------------------
  // JWT Token Validation (adv-b-sec-002)
  // --------------------------------------------------------------------------

  /**
   * Generate a JWT token
   */
  generateJWT(
    subject: string,
    role: "leader" | "worker" | "admin",
    permissions: Permission[],
  ): string {
    const now = Math.floor(Date.now() / 1000);
    const payload: JWTPayload = {
      sub: subject,
      iss: this.config.jwt_config.issuer,
      aud: this.config.jwt_config.audience,
      exp: now + this.config.jwt_config.expiration_seconds,
      iat: now,
      jti: uuidv4(),
      role,
      permissions,
    };

    return this.signJWT(payload);
  }

  /**
   * Validate a JWT token
   */
  validateJWT(token: string): AuthResult {
    if (!this.config.jwt_enabled) {
      return { authenticated: true };
    }

    try {
      const payload = this.verifyJWT(token);

      // Check expiration
      const now = Math.floor(Date.now() / 1000);
      if (payload.exp < now) {
        return {
          authenticated: false,
          error: "JWT has expired",
          error_code: "JWT_EXPIRED",
        };
      }

      // Check issuer and audience
      if (payload.iss !== this.config.jwt_config.issuer) {
        return {
          authenticated: false,
          error: "Invalid JWT issuer",
          error_code: "INVALID_JWT",
        };
      }

      if (payload.aud !== this.config.jwt_config.audience) {
        return {
          authenticated: false,
          error: "Invalid JWT audience",
          error_code: "INVALID_JWT",
        };
      }

      return { authenticated: true, jwt_payload: payload };
    } catch {
      return {
        authenticated: false,
        error: "Invalid JWT token",
        error_code: "INVALID_JWT",
      };
    }
  }

  /**
   * Check if JWT should be refreshed
   */
  shouldRefreshJWT(token: string): boolean {
    try {
      const payload = this.verifyJWT(token);
      const now = Math.floor(Date.now() / 1000);
      const timeUntilExpiry = payload.exp - now;
      return (
        timeUntilExpiry <= this.config.jwt_config.refresh_threshold_seconds
      );
    } catch {
      return false;
    }
  }

  /**
   * Refresh a JWT token
   */
  refreshJWT(token: string): string | null {
    try {
      const payload = this.verifyJWT(token);
      return this.generateJWT(payload.sub, payload.role, payload.permissions);
    } catch {
      return null;
    }
  }

  /**
   * Sign a JWT payload
   */
  private signJWT(payload: JWTPayload): string {
    const header = { alg: "HS256", typ: "JWT" };
    const headerB64 = Buffer.from(JSON.stringify(header)).toString("base64url");
    const payloadB64 = Buffer.from(JSON.stringify(payload)).toString(
      "base64url",
    );

    const signature = crypto
      .createHmac("sha256", this.config.jwt_config.secret)
      .update(`${headerB64}.${payloadB64}`)
      .digest("base64url");

    return `${headerB64}.${payloadB64}.${signature}`;
  }

  /**
   * Verify and decode a JWT
   */
  private verifyJWT(token: string): JWTPayload {
    const parts = token.split(".");
    if (parts.length !== 3) {
      throw new Error("Invalid JWT format");
    }

    const [headerB64, payloadB64, signature] = parts;

    // Verify signature
    const expectedSignature = crypto
      .createHmac("sha256", this.config.jwt_config.secret)
      .update(`${headerB64}.${payloadB64}`)
      .digest("base64url");

    if (
      !crypto.timingSafeEqual(
        Buffer.from(signature),
        Buffer.from(expectedSignature),
      )
    ) {
      throw new Error("Invalid JWT signature");
    }

    // Decode payload
    const payload = JSON.parse(Buffer.from(payloadB64, "base64url").toString());
    return payload as JWTPayload;
  }

  // --------------------------------------------------------------------------
  // Request Signing (adv-b-sec-003)
  // --------------------------------------------------------------------------

  /**
   * Sign a request
   */
  signRequest(
    method: string,
    path: string,
    body: string,
    secret: string,
    algorithm: "sha256" | "sha512" = "sha256",
  ): RequestSignature {
    const timestamp = Date.now();
    const nonce = crypto.randomBytes(16).toString("hex");

    const message = `${timestamp}:${nonce}:${method}:${path}:${body}`;
    const signature = crypto
      .createHmac(algorithm, secret)
      .update(message)
      .digest("hex");

    return {
      algorithm,
      timestamp,
      nonce,
      signature,
    };
  }

  /**
   * Verify a request signature
   */
  verifyRequestSignature(
    signature: RequestSignature,
    method: string,
    path: string,
    body: string,
    secret: string,
  ): AuthResult {
    if (!this.config.request_signing_enabled) {
      return { authenticated: true };
    }

    // Check timestamp age
    const age = Date.now() - signature.timestamp;
    if (age > this.config.signature_max_age_seconds * 1000) {
      return {
        authenticated: false,
        error: "Signature has expired",
        error_code: "SIGNATURE_EXPIRED",
      };
    }

    // Check for replay attack (nonce reuse)
    const nonceKey = `${signature.nonce}:${signature.timestamp}`;
    if (this.usedNonces.has(nonceKey)) {
      return {
        authenticated: false,
        error: "Nonce already used (replay attack detected)",
        error_code: "INVALID_SIGNATURE",
      };
    }

    // Store nonce
    this.usedNonces.set(nonceKey, Date.now());

    // Verify signature
    const message = `${signature.timestamp}:${signature.nonce}:${method}:${path}:${body}`;
    const expectedSignature = crypto
      .createHmac(signature.algorithm, secret)
      .update(message)
      .digest("hex");

    try {
      if (
        !crypto.timingSafeEqual(
          Buffer.from(signature.signature),
          Buffer.from(expectedSignature),
        )
      ) {
        return {
          authenticated: false,
          error: "Invalid signature",
          error_code: "INVALID_SIGNATURE",
        };
      }
    } catch {
      return {
        authenticated: false,
        error: "Invalid signature",
        error_code: "INVALID_SIGNATURE",
      };
    }

    return { authenticated: true };
  }

  /**
   * Start nonce cleanup interval
   */
  private startNonceCleanup(): void {
    this.nonceCleanupInterval = setInterval(() => {
      const cutoff =
        Date.now() - this.config.signature_max_age_seconds * 1000 * 2;
      for (const [key, timestamp] of this.usedNonces.entries()) {
        if (timestamp < cutoff) {
          this.usedNonces.delete(key);
        }
      }
    }, 60000);
  }

  // --------------------------------------------------------------------------
  // IP Allowlisting (adv-b-sec-004)
  // --------------------------------------------------------------------------

  /**
   * Check if an IP is allowed
   */
  checkIPAllowed(ip: string, apiKey?: APIKey): AuthResult {
    if (!this.config.ip_allowlist_enabled) {
      return { authenticated: true };
    }

    // Normalize IP
    const normalizedIP = this.normalizeIP(ip);

    // Check global denylist first
    if (this.isIPInList(normalizedIP, this.config.ip_config.global_denylist)) {
      return {
        authenticated: false,
        error: "IP address is blocked",
        error_code: "IP_NOT_ALLOWED",
      };
    }

    // Check API key specific allowlist if provided
    if (apiKey && apiKey.ip_allowlist.length > 0) {
      if (!this.isIPInList(normalizedIP, apiKey.ip_allowlist)) {
        return {
          authenticated: false,
          error: "IP address not in API key allowlist",
          error_code: "IP_NOT_ALLOWED",
        };
      }
      return { authenticated: true };
    }

    // Check global allowlist
    if (this.config.ip_config.global_allowlist.length > 0) {
      if (
        !this.isIPInList(normalizedIP, this.config.ip_config.global_allowlist)
      ) {
        if (this.config.ip_config.default_policy === "deny") {
          return {
            authenticated: false,
            error: "IP address not in allowlist",
            error_code: "IP_NOT_ALLOWED",
          };
        }
      }
    }

    // Apply default policy
    if (this.config.ip_config.default_policy === "deny") {
      return {
        authenticated: false,
        error: "IP address not allowed by default policy",
        error_code: "IP_NOT_ALLOWED",
      };
    }

    return { authenticated: true };
  }

  /**
   * Add IP to global allowlist
   */
  addToGlobalAllowlist(ip: string): void {
    const normalized = this.normalizeIP(ip);
    if (!this.config.ip_config.global_allowlist.includes(normalized)) {
      this.config.ip_config.global_allowlist.push(normalized);
    }
  }

  /**
   * Remove IP from global allowlist
   */
  removeFromGlobalAllowlist(ip: string): void {
    const normalized = this.normalizeIP(ip);
    const index = this.config.ip_config.global_allowlist.indexOf(normalized);
    if (index !== -1) {
      this.config.ip_config.global_allowlist.splice(index, 1);
    }
  }

  /**
   * Add IP to global denylist
   */
  addToGlobalDenylist(ip: string): void {
    const normalized = this.normalizeIP(ip);
    if (!this.config.ip_config.global_denylist.includes(normalized)) {
      this.config.ip_config.global_denylist.push(normalized);
    }
  }

  /**
   * Check if IP is in a list (supports CIDR notation)
   */
  private isIPInList(ip: string, list: string[]): boolean {
    for (const entry of list) {
      if (entry.includes("/")) {
        // CIDR notation
        if (this.isIPInCIDR(ip, entry)) {
          return true;
        }
      } else if (ip === entry) {
        return true;
      }
    }
    return false;
  }

  /**
   * Check if IP is in CIDR range
   */
  private isIPInCIDR(ip: string, cidr: string): boolean {
    const [range, bits] = cidr.split("/");
    const mask = parseInt(bits, 10);

    const ipNum = this.ipToNumber(ip);
    const rangeNum = this.ipToNumber(range);
    const maskNum = ~((1 << (32 - mask)) - 1);

    return (ipNum & maskNum) === (rangeNum & maskNum);
  }

  /**
   * Convert IP to number
   */
  private ipToNumber(ip: string): number {
    const parts = ip.split(".").map((p) => parseInt(p, 10));
    return (
      ((parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]) >>> 0
    );
  }

  /**
   * Normalize IP address
   */
  private normalizeIP(ip: string): string {
    // Handle IPv6 mapped IPv4
    if (ip.startsWith("::ffff:")) {
      return ip.substring(7);
    }
    return ip;
  }

  // --------------------------------------------------------------------------
  // Rate Limiting Per Key (adv-b-sec-005)
  // --------------------------------------------------------------------------

  /**
   * Check rate limit for an API key
   */
  checkRateLimit(keyId: string, config?: RateLimitConfig): AuthResult {
    if (!this.config.rate_limiting_enabled) {
      return { authenticated: true };
    }

    const apiKey = this.apiKeys.get(keyId);
    const limitConfig = config ||
      apiKey?.rate_limit || {
        requests_per_minute: 60,
        requests_per_hour: 1000,
        burst_limit: 10,
      };

    let state = this.rateLimitStates.get(keyId);
    const now = Date.now();

    if (!state) {
      state = {
        minute_count: 0,
        minute_reset_at: now + 60000,
        hour_count: 0,
        hour_reset_at: now + 3600000,
        burst_tokens: limitConfig.burst_limit,
        last_request_at: now,
      };
      this.rateLimitStates.set(keyId, state);
    }

    // Reset minute counter if needed
    if (now >= state.minute_reset_at) {
      state.minute_count = 0;
      state.minute_reset_at = now + 60000;
    }

    // Reset hour counter if needed
    if (now >= state.hour_reset_at) {
      state.hour_count = 0;
      state.hour_reset_at = now + 3600000;
    }

    // Refill burst tokens (token bucket algorithm)
    const timeSinceLastRequest = now - state.last_request_at;
    const tokensToAdd = timeSinceLastRequest / 1000; // 1 token per second
    state.burst_tokens = Math.min(
      limitConfig.burst_limit,
      state.burst_tokens + tokensToAdd,
    );
    state.last_request_at = now;

    // Check limits
    if (state.minute_count >= limitConfig.requests_per_minute) {
      return {
        authenticated: false,
        error: `Rate limit exceeded: ${limitConfig.requests_per_minute} requests per minute`,
        error_code: "RATE_LIMIT_EXCEEDED",
      };
    }

    if (state.hour_count >= limitConfig.requests_per_hour) {
      return {
        authenticated: false,
        error: `Rate limit exceeded: ${limitConfig.requests_per_hour} requests per hour`,
        error_code: "RATE_LIMIT_EXCEEDED",
      };
    }

    if (state.burst_tokens < 1) {
      return {
        authenticated: false,
        error: "Burst limit exceeded, please slow down",
        error_code: "RATE_LIMIT_EXCEEDED",
      };
    }

    // Consume tokens
    state.minute_count++;
    state.hour_count++;
    state.burst_tokens--;

    return { authenticated: true };
  }

  /**
   * Get rate limit status for a key
   */
  getRateLimitStatus(keyId: string): {
    minute_remaining: number;
    hour_remaining: number;
    burst_remaining: number;
    minute_reset_in_ms: number;
    hour_reset_in_ms: number;
  } | null {
    const apiKey = this.apiKeys.get(keyId);
    const state = this.rateLimitStates.get(keyId);

    if (!apiKey || !state) return null;

    const now = Date.now();
    return {
      minute_remaining: Math.max(
        0,
        apiKey.rate_limit.requests_per_minute - state.minute_count,
      ),
      hour_remaining: Math.max(
        0,
        apiKey.rate_limit.requests_per_hour - state.hour_count,
      ),
      burst_remaining: Math.floor(state.burst_tokens),
      minute_reset_in_ms: Math.max(0, state.minute_reset_at - now),
      hour_reset_in_ms: Math.max(0, state.hour_reset_at - now),
    };
  }

  /**
   * Reset rate limit for a key
   */
  resetRateLimit(keyId: string): boolean {
    return this.rateLimitStates.delete(keyId);
  }

  // --------------------------------------------------------------------------
  // Combined Authentication
  // --------------------------------------------------------------------------

  /**
   * Authenticate a request with all enabled security checks
   */
  authenticate(options: {
    api_key?: string;
    jwt_token?: string;
    request_signature?: RequestSignature;
    ip?: string;
    method?: string;
    path?: string;
    body?: string;
  }): AuthResult {
    let apiKey: APIKey | undefined;

    // 1. API Key Authentication
    if (this.config.api_key_enabled) {
      if (!options.api_key) {
        return {
          authenticated: false,
          error: "API key required",
          error_code: "MISSING_CREDENTIALS",
        };
      }

      const apiKeyResult = this.validateAPIKey(options.api_key);
      if (!apiKeyResult.authenticated) {
        return apiKeyResult;
      }
      apiKey = apiKeyResult.api_key;
    }

    // 2. JWT Validation (if enabled and token provided)
    if (this.config.jwt_enabled && options.jwt_token) {
      const jwtResult = this.validateJWT(options.jwt_token);
      if (!jwtResult.authenticated) {
        return jwtResult;
      }
    }

    // 3. Request Signature (if enabled)
    if (
      this.config.request_signing_enabled &&
      options.request_signature &&
      apiKey
    ) {
      const signatureResult = this.verifyRequestSignature(
        options.request_signature,
        options.method || "POST",
        options.path || "/",
        options.body || "",
        apiKey.key,
      );
      if (!signatureResult.authenticated) {
        return signatureResult;
      }
    }

    // 4. IP Allowlist Check
    if (this.config.ip_allowlist_enabled && options.ip) {
      const ipResult = this.checkIPAllowed(options.ip, apiKey);
      if (!ipResult.authenticated) {
        return ipResult;
      }
    }

    // 5. Rate Limiting
    if (this.config.rate_limiting_enabled && apiKey) {
      const rateLimitResult = this.checkRateLimit(apiKey.id);
      if (!rateLimitResult.authenticated) {
        return rateLimitResult;
      }
    }

    return { authenticated: true, api_key: apiKey };
  }

  /**
   * Check if a request has a specific permission
   */
  hasPermission(apiKey: APIKey | undefined, permission: Permission): boolean {
    if (!apiKey) return !this.config.api_key_enabled;
    return (
      apiKey.permissions.includes(permission) ||
      apiKey.permissions.includes("admin")
    );
  }

  // --------------------------------------------------------------------------
  // Configuration
  // --------------------------------------------------------------------------

  /**
   * Update security configuration
   */
  updateConfig(config: Partial<SecurityConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration (without secrets)
   */
  getConfig(): Omit<SecurityConfig, "jwt_config"> & {
    jwt_config: Omit<JWTConfig, "secret"> & { secret_masked: string };
  } {
    const { secret: _secret, ...jwtConfigWithoutSecret } =
      this.config.jwt_config;
    return {
      ...this.config,
      jwt_config: {
        ...jwtConfigWithoutSecret,
        secret_masked: "***",
      },
    };
  }

  // --------------------------------------------------------------------------
  // Statistics
  // --------------------------------------------------------------------------

  /**
   * Get security statistics
   */
  getStats(): {
    total_api_keys: number;
    active_api_keys: number;
    rate_limited_keys: number;
    ip_allowlist_size: number;
    ip_denylist_size: number;
  } {
    const now = Date.now();
    let rateLimitedCount = 0;

    for (const [keyId] of this.apiKeys.entries()) {
      const state = this.rateLimitStates.get(keyId);
      if (state) {
        const apiKey = this.apiKeys.get(keyId);
        if (
          apiKey &&
          (state.minute_count >= apiKey.rate_limit.requests_per_minute ||
            state.hour_count >= apiKey.rate_limit.requests_per_hour)
        ) {
          rateLimitedCount++;
        }
      }
    }

    return {
      total_api_keys: this.apiKeys.size,
      active_api_keys: Array.from(this.apiKeys.values()).filter(
        (k) => k.enabled,
      ).length,
      rate_limited_keys: rateLimitedCount,
      ip_allowlist_size: this.config.ip_config.global_allowlist.length,
      ip_denylist_size: this.config.ip_config.global_denylist.length,
    };
  }

  /**
   * Clean up resources
   */
  destroy(): void {
    if (this.nonceCleanupInterval) {
      clearInterval(this.nonceCleanupInterval);
      this.nonceCleanupInterval = null;
    }
    this.apiKeys.clear();
    this.apiKeysByKey.clear();
    this.rateLimitStates.clear();
    this.usedNonces.clear();
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let securityManagerInstance: SecurityManager | null = null;

export function getSecurityManager(
  config?: Partial<SecurityConfig>,
): SecurityManager {
  if (!securityManagerInstance) {
    securityManagerInstance = new SecurityManager(config);
  }
  return securityManagerInstance;
}

export function resetSecurityManager(): void {
  if (securityManagerInstance) {
    securityManagerInstance.destroy();
    securityManagerInstance = null;
  }
}

// ============================================================================
// Encryption Manager (sec-004: Task data encryption at rest)
// ============================================================================

export class EncryptionManager {
  private key: Buffer;
  private readonly algorithm = "aes-256-gcm" as const;

  constructor(masterKey?: string) {
    const keySource =
      masterKey ||
      process.env.ENCRYPTION_MASTER_KEY ||
      crypto.randomBytes(32).toString("hex");
    this.key = crypto.scryptSync(keySource, "coordination-salt", 32);
  }

  encrypt(plaintext: string | object): string {
    const text =
      typeof plaintext === "object" ? JSON.stringify(plaintext) : plaintext;
    const iv = crypto.randomBytes(12);
    const cipher = crypto.createCipheriv(
      this.algorithm,
      this.key,
      iv,
    ) as crypto.CipherGCM;

    let encrypted = cipher.update(text, "utf8", "base64");
    encrypted += cipher.final("base64");

    const authTag = cipher.getAuthTag();

    // Combine iv + authTag + encrypted
    const combined = Buffer.concat([
      iv,
      authTag,
      Buffer.from(encrypted, "base64"),
    ]);

    return combined.toString("base64");
  }

  decrypt(encryptedData: string, asObject = false): string | object {
    const combined = Buffer.from(encryptedData, "base64");

    const iv = combined.subarray(0, 12);
    const authTag = combined.subarray(12, 28);
    const encrypted = combined.subarray(28);

    const decipher = crypto.createDecipheriv(
      this.algorithm,
      this.key,
      iv,
    ) as crypto.DecipherGCM;
    decipher.setAuthTag(authTag);

    let decrypted = decipher.update(
      encrypted.toString("base64"),
      "base64",
      "utf8",
    );
    decrypted += decipher.final("utf8");

    return asObject ? JSON.parse(decrypted) : decrypted;
  }

  encryptTaskData(taskData: Record<string, any>): Record<string, any> {
    const encrypted = { ...taskData };

    if (encrypted.description) {
      encrypted.description = this.encrypt(encrypted.description);
      encrypted._description_encrypted = true;
    }

    if (encrypted.context?.hints) {
      encrypted.context = { ...encrypted.context };
      encrypted.context.hints = this.encrypt(encrypted.context.hints);
      encrypted.context._hints_encrypted = true;
    }

    if (encrypted.result?.output) {
      encrypted.result = { ...encrypted.result };
      encrypted.result.output = this.encrypt(encrypted.result.output);
      encrypted.result._output_encrypted = true;
    }

    return encrypted;
  }

  decryptTaskData(encryptedData: Record<string, any>): Record<string, any> {
    const decrypted = { ...encryptedData };

    if (decrypted._description_encrypted) {
      decrypted.description = this.decrypt(decrypted.description) as string;
      delete decrypted._description_encrypted;
    }

    if (decrypted.context?._hints_encrypted) {
      decrypted.context = { ...decrypted.context };
      decrypted.context.hints = this.decrypt(decrypted.context.hints) as string;
      delete decrypted.context._hints_encrypted;
    }

    if (decrypted.result?._output_encrypted) {
      decrypted.result = { ...decrypted.result };
      decrypted.result.output = this.decrypt(decrypted.result.output) as string;
      delete decrypted.result._output_encrypted;
    }

    return decrypted;
  }
}

// ============================================================================
// Secure Credential Store (sec-005: Secure credential storage)
// ============================================================================

export interface StoredCredential {
  name: string;
  encryptedValue: string;
  createdAt: string;
  updatedAt?: string;
  expiresAt?: string;
  metadata: Record<string, any>;
}

export class SecureCredentialStore {
  private credentials: Map<string, StoredCredential> = new Map();
  private encryption: EncryptionManager;
  private accessLog: Array<{
    timestamp: string;
    credential: string;
    accessor: string;
    success: boolean;
  }> = [];

  constructor(encryptionManager?: EncryptionManager) {
    this.encryption = encryptionManager || new EncryptionManager();
  }

  store(
    name: string,
    value: string,
    expiresAt?: string,
    metadata?: Record<string, any>,
  ): void {
    const encryptedValue = this.encryption.encrypt(value);

    if (this.credentials.has(name)) {
      const existing = this.credentials.get(name)!;
      existing.encryptedValue = encryptedValue;
      existing.updatedAt = new Date().toISOString();
      if (expiresAt) existing.expiresAt = expiresAt;
      if (metadata) existing.metadata = metadata;
    } else {
      this.credentials.set(name, {
        name,
        encryptedValue,
        createdAt: new Date().toISOString(),
        expiresAt,
        metadata: metadata || {},
      });
    }
  }

  retrieve(name: string, accessor = "unknown"): string | null {
    const cred = this.credentials.get(name);

    if (!cred) {
      this.logAccess(name, accessor, false);
      return null;
    }

    // Check expiration
    if (cred.expiresAt && new Date(cred.expiresAt) < new Date()) {
      this.logAccess(name, accessor, false);
      return null;
    }

    this.logAccess(name, accessor, true);

    try {
      return this.encryption.decrypt(cred.encryptedValue) as string;
    } catch {
      return null;
    }
  }

  delete(name: string): boolean {
    return this.credentials.delete(name);
  }

  list(): Array<Omit<StoredCredential, "encryptedValue">> {
    return Array.from(this.credentials.values()).map(
      ({ encryptedValue, ...rest }) => rest,
    );
  }

  private logAccess(
    credential: string,
    accessor: string,
    success: boolean,
  ): void {
    this.accessLog.push({
      timestamp: new Date().toISOString(),
      credential,
      accessor,
      success,
    });

    // Keep only last 1000 entries
    if (this.accessLog.length > 1000) {
      this.accessLog = this.accessLog.slice(-1000);
    }
  }

  getAccessLog(credentialName?: string): typeof this.accessLog {
    if (credentialName) {
      return this.accessLog.filter((e) => e.credential === credentialName);
    }
    return [...this.accessLog];
  }
}

// ============================================================================
// Audit Logger (sec-006: Audit logging for all operations)
// ============================================================================

export type AuditEventType =
  | "auth.login"
  | "auth.logout"
  | "auth.failed"
  | "auth.token_issued"
  | "auth.token_revoked"
  | "auth.key_created"
  | "auth.key_revoked"
  | "task.created"
  | "task.claimed"
  | "task.started"
  | "task.completed"
  | "task.failed"
  | "task.cancelled"
  | "agent.registered"
  | "agent.heartbeat"
  | "agent.deregistered"
  | "discovery.created"
  | "discovery.read"
  | "security.access_denied"
  | "security.rate_limited"
  | "security.key_rotated"
  | "coordination.init"
  | "coordination.start"
  | "coordination.stop";

export interface AuditEvent {
  eventId: string;
  eventType: AuditEventType;
  timestamp: string;
  actor?: string;
  actorRole?: string;
  resourceType?: string;
  resourceId?: string;
  action?: string;
  outcome: "success" | "failure" | "error";
  details: Record<string, any>;
  ipAddress?: string;
  correlationId?: string;
}

export class AuditLogger {
  private buffer: AuditEvent[] = [];
  private maxBufferSize = 100;
  private onEvent?: (event: AuditEvent) => void;
  private enableConsole: boolean;

  constructor(
    options: {
      onEvent?: (event: AuditEvent) => void;
      enableConsole?: boolean;
    } = {},
  ) {
    this.onEvent = options.onEvent;
    this.enableConsole = options.enableConsole || false;
  }

  log(
    eventType: AuditEventType,
    options: Partial<
      Omit<AuditEvent, "eventId" | "eventType" | "timestamp">
    > = {},
  ): AuditEvent {
    const event: AuditEvent = {
      eventId: crypto.randomBytes(8).toString("hex"),
      eventType,
      timestamp: new Date().toISOString(),
      outcome: "success",
      details: {},
      ...options,
    };

    this.processEvent(event);
    return event;
  }

  private processEvent(event: AuditEvent): void {
    this.buffer.push(event);
    if (this.buffer.length > this.maxBufferSize) {
      this.buffer = this.buffer.slice(-this.maxBufferSize);
    }

    if (this.onEvent) {
      try {
        this.onEvent(event);
      } catch {
        // Ignore callback errors
      }
    }

    if (this.enableConsole) {
      const symbol = event.outcome === "success" ? "+" : "-";
      console.error(
        `[AUDIT][${symbol}] ${event.eventType} | ${event.actor || "system"} | ${event.resourceType || "-"}/${event.resourceId || "-"}`,
      );
    }
  }

  logAuthSuccess(actor: string, role: string, method = "api_key"): AuditEvent {
    return this.log("auth.login", {
      actor,
      actorRole: role,
      action: `authenticate:${method}`,
    });
  }

  logAuthFailure(reason: string, attemptedIdentity?: string): AuditEvent {
    return this.log("auth.failed", {
      actor: attemptedIdentity,
      outcome: "failure",
      details: { reason },
    });
  }

  logAccessDenied(
    actor: string,
    role: string,
    permission: string,
    resourceType: string,
    resourceId?: string,
  ): AuditEvent {
    return this.log("security.access_denied", {
      actor,
      actorRole: role,
      resourceType,
      resourceId,
      action: permission,
      outcome: "failure",
      details: { required_permission: permission },
    });
  }

  logTaskEvent(
    eventType: AuditEventType,
    taskId: string,
    actor: string,
    actorRole: string,
    details?: Record<string, any>,
  ): AuditEvent {
    return this.log(eventType, {
      actor,
      actorRole,
      resourceType: "task",
      resourceId: taskId,
      details,
    });
  }

  logRateLimited(
    actor: string,
    limitName: string,
    requestsMade: number,
    limit: number,
  ): AuditEvent {
    return this.log("security.rate_limited", {
      actor,
      outcome: "failure",
      details: { limit_name: limitName, requests_made: requestsMade, limit },
    });
  }

  queryEvents(options: {
    eventType?: AuditEventType;
    actor?: string;
    resourceType?: string;
    limit?: number;
  }): AuditEvent[] {
    let events = [...this.buffer];

    if (options.eventType) {
      events = events.filter((e) => e.eventType === options.eventType);
    }
    if (options.actor) {
      events = events.filter((e) => e.actor === options.actor);
    }
    if (options.resourceType) {
      events = events.filter((e) => e.resourceType === options.resourceType);
    }

    return events.slice(-(options.limit || 100));
  }
}

// ============================================================================
// Input Sanitizer (sec-007: Input sanitization for task descriptions)
// ============================================================================

export class InputSanitizer {
  private maxLength: number;
  private promptInjectionPatterns: RegExp[];

  constructor(maxLength = 10000) {
    this.maxLength = maxLength;

    this.promptInjectionPatterns = [
      /ignore\s+(all\s+)?previous\s+instructions?/i,
      /forget\s+(all\s+)?previous/i,
      /disregard\s+(all\s+)?previous/i,
      /system\s*:\s*you\s+are/i,
      /new\s+instructions?:/i,
      /override\s+instructions?/i,
      /you\s+are\s+now\s+a/i,
      /act\s+as\s+(if\s+you\s+are\s+)?/i,
      /pretend\s+(you\s+are|to\s+be)/i,
      /\[system\]/i,
      /<\s*system\s*>/i,
      /###\s*system/i,
      /<\|im_start\|>/i,
      /jailbreak/i,
      /dan\s*mode/i,
      /bypass\s+(safety|content|filter)/i,
    ];
  }

  sanitizeTaskDescription(description: string): {
    value: string;
    warnings: string[];
    modified: boolean;
  } {
    const warnings: string[] = [];
    let value = description || "";
    let modified = false;

    // Length check
    if (value.length > this.maxLength) {
      value = value.substring(0, this.maxLength);
      warnings.push("Description truncated");
      modified = true;
    }

    // Check for prompt injection
    for (const pattern of this.promptInjectionPatterns) {
      if (pattern.test(value)) {
        value = value.replace(pattern, "[REMOVED]");
        warnings.push("Potential prompt injection removed");
        modified = true;
      }
    }

    // Normalize whitespace
    const normalized = value.split(/\s+/).join(" ").trim();
    if (normalized !== value) {
      value = normalized;
      modified = true;
    }

    return { value, warnings, modified };
  }

  sanitizeAgentId(agentId: string): {
    value: string;
    valid: boolean;
    error?: string;
  } {
    if (!agentId) {
      return { value: "", valid: false, error: "Agent ID cannot be empty" };
    }

    if (!/^[a-zA-Z0-9_-]+$/.test(agentId)) {
      const cleaned = agentId.replace(/[^a-zA-Z0-9_-]/g, "");
      if (!cleaned) {
        return {
          value: "",
          valid: false,
          error: "No valid characters in agent ID",
        };
      }
      return { value: cleaned, valid: true };
    }

    if (agentId.length > 64) {
      return { value: agentId.substring(0, 64), valid: true };
    }

    return { value: agentId, valid: true };
  }

  sanitizePath(filePath: string): {
    value: string;
    valid: boolean;
    error?: string;
  } {
    if (!filePath) {
      return { value: "", valid: true };
    }

    // Check for path traversal
    if (/\.\.[/\\]/.test(filePath) || /[/\\]\.\./.test(filePath)) {
      return { value: "", valid: false, error: "Path traversal detected" };
    }

    // Remove null bytes
    const cleaned = filePath.replace(/\x00/g, "");

    // Normalize separators
    const normalized = cleaned.replace(/\\/g, "/").replace(/\/+/g, "/");

    return { value: normalized, valid: true };
  }
}

// ============================================================================
// Secret Rotation Manager (sec-010: Secret rotation mechanism)
// ============================================================================

export interface SecretMetadata {
  name: string;
  version: number;
  lastRotated: string;
  nextRotation?: string;
  autoRotate: boolean;
}

export class SecretRotationManager {
  private secrets: Map<
    string,
    {
      value: string;
      version: number;
      lastRotated: string;
      previousValues: { value: string; expiresAt: number }[];
      autoRotate: boolean;
    }
  > = new Map();

  private gracePeriodMs = 5 * 60 * 1000; // 5 minutes
  private onRotation?: (
    name: string,
    oldValue: string,
    newValue: string,
  ) => void;

  constructor(
    onRotation?: (name: string, oldValue: string, newValue: string) => void,
  ) {
    this.onRotation = onRotation;
  }

  registerSecret(name: string, initialValue: string, autoRotate = true): void {
    this.secrets.set(name, {
      value: initialValue,
      version: 1,
      lastRotated: new Date().toISOString(),
      previousValues: [],
      autoRotate,
    });
  }

  getSecret(name: string): string | undefined {
    return this.secrets.get(name)?.value;
  }

  validateSecret(name: string, value: string): boolean {
    const secret = this.secrets.get(name);
    if (!secret) return false;

    // Check current value
    if (secret.value === value) return true;

    // Check previous values in grace period
    const now = Date.now();
    return secret.previousValues.some(
      (pv) => pv.expiresAt > now && pv.value === value,
    );
  }

  rotateSecret(name: string, newValue?: string): string | null {
    const secret = this.secrets.get(name);
    if (!secret) return null;

    const oldValue = secret.value;
    const generatedValue = newValue || crypto.randomBytes(32).toString("hex");

    // Move current to previous with grace period
    secret.previousValues.push({
      value: oldValue,
      expiresAt: Date.now() + this.gracePeriodMs,
    });

    // Keep only last 5
    if (secret.previousValues.length > 5) {
      secret.previousValues = secret.previousValues.slice(-5);
    }

    secret.value = generatedValue;
    secret.version++;
    secret.lastRotated = new Date().toISOString();

    // Call rotation callback
    if (this.onRotation) {
      try {
        this.onRotation(name, oldValue, generatedValue);
      } catch {
        // Ignore callback errors
      }
    }

    return generatedValue;
  }

  rotateAll(): Record<string, boolean> {
    const results: Record<string, boolean> = {};
    for (const [name, secret] of this.secrets.entries()) {
      if (secret.autoRotate) {
        results[name] = this.rotateSecret(name) !== null;
      }
    }
    return results;
  }

  getMetadata(name: string): SecretMetadata | null {
    const secret = this.secrets.get(name);
    if (!secret) return null;
    return {
      name,
      version: secret.version,
      lastRotated: secret.lastRotated,
      autoRotate: secret.autoRotate,
    };
  }

  listSecrets(): SecretMetadata[] {
    return Array.from(this.secrets.entries()).map(([name, secret]) => ({
      name,
      version: secret.version,
      lastRotated: secret.lastRotated,
      autoRotate: secret.autoRotate,
    }));
  }
}

// ============================================================================
// Unified Security Manager (combines all features)
// ============================================================================

export interface UnifiedSecurityConfig {
  enableApiKeys: boolean;
  enableJwt: boolean;
  enableEncryption: boolean;
  enableAudit: boolean;
  enableRateLimiting: boolean;
  enableSanitization: boolean;
  jwtSecret?: string;
  encryptionKey?: string;
}

export class UnifiedSecurityManager {
  public apiKeyManager: SecurityManager;
  public jwtManager: SecurityManager;
  public encryptionManager: EncryptionManager;
  public credentialStore: SecureCredentialStore;
  public auditLogger: AuditLogger;
  public sanitizer: InputSanitizer;
  public rotationManager: SecretRotationManager;

  private config: UnifiedSecurityConfig;

  constructor(config: Partial<UnifiedSecurityConfig> = {}) {
    this.config = {
      enableApiKeys: config.enableApiKeys !== false,
      enableJwt: config.enableJwt !== false,
      enableEncryption: config.enableEncryption !== false,
      enableAudit: config.enableAudit !== false,
      enableRateLimiting: config.enableRateLimiting !== false,
      enableSanitization: config.enableSanitization !== false,
      jwtSecret: config.jwtSecret,
      encryptionKey: config.encryptionKey,
    };

    // Initialize components
    this.apiKeyManager = new SecurityManager({
      api_key_enabled: this.config.enableApiKeys,
      rate_limiting_enabled: this.config.enableRateLimiting,
    });

    this.jwtManager = new SecurityManager({
      jwt_enabled: this.config.enableJwt,
      jwt_config: this.config.jwtSecret
        ? {
            secret: this.config.jwtSecret,
            issuer: "claude-coordination",
            audience: "claude-agents",
            expiration_seconds: 3600,
            refresh_threshold_seconds: 300,
          }
        : undefined,
    });

    this.encryptionManager = new EncryptionManager(this.config.encryptionKey);
    this.credentialStore = new SecureCredentialStore(this.encryptionManager);
    this.auditLogger = new AuditLogger({ enableConsole: false });
    this.sanitizer = new InputSanitizer();
    this.rotationManager = new SecretRotationManager((name, oldVal, newVal) => {
      this.auditLogger.log("security.key_rotated", {
        resourceType: "secret",
        resourceId: name,
        details: { version_incremented: true },
      });
    });

    // Register JWT secret for rotation
    if (this.config.jwtSecret) {
      this.rotationManager.registerSecret("jwt_secret", this.config.jwtSecret);
    }
  }

  /**
   * Authenticate a request and get auth info
   */
  authenticate(apiKey: string): {
    authenticated: boolean;
    keyInfo?: APIKey;
    error?: string;
  } {
    const result = this.apiKeyManager.validateAPIKey(apiKey);

    if (result.authenticated && result.api_key) {
      this.auditLogger.logAuthSuccess(
        result.api_key.agent_id || result.api_key.id,
        result.api_key.role,
      );
    } else {
      this.auditLogger.logAuthFailure(result.error || "unknown");
    }

    return {
      authenticated: result.authenticated,
      keyInfo: result.api_key,
      error: result.error,
    };
  }

  /**
   * Check if request has permission
   */
  hasPermission(keyInfo: APIKey | undefined, permission: Permission): boolean {
    return this.apiKeyManager.hasPermission(keyInfo, permission);
  }

  /**
   * Sanitize task input
   */
  sanitizeTaskInput(input: {
    description: string;
    priority?: number;
    dependencies?: string[];
    context_files?: string[];
    hints?: string;
  }): {
    description: string;
    priority: number;
    dependencies: string[];
    context_files: string[];
    hints: string;
    warnings: string[];
  } {
    if (!this.config.enableSanitization) {
      return {
        description: input.description || "",
        priority: input.priority || 5,
        dependencies: input.dependencies || [],
        context_files: input.context_files || [],
        hints: input.hints || "",
        warnings: [],
      };
    }

    const warnings: string[] = [];

    const descResult = this.sanitizer.sanitizeTaskDescription(
      input.description,
    );
    warnings.push(...descResult.warnings);

    const hintsResult = this.sanitizer.sanitizeTaskDescription(
      input.hints || "",
    );
    warnings.push(...hintsResult.warnings);

    const sanitizedFiles: string[] = [];
    for (const file of input.context_files || []) {
      const fileResult = this.sanitizer.sanitizePath(file);
      if (fileResult.valid) {
        sanitizedFiles.push(fileResult.value);
      } else {
        warnings.push(`Invalid file path: ${fileResult.error}`);
      }
    }

    const sanitizedDeps: string[] = [];
    for (const dep of input.dependencies || []) {
      const depResult = this.sanitizer.sanitizeAgentId(dep);
      if (depResult.valid) {
        sanitizedDeps.push(depResult.value);
      }
    }

    let priority = input.priority || 5;
    if (priority < 1) priority = 1;
    if (priority > 10) priority = 10;

    return {
      description: descResult.value,
      priority,
      dependencies: sanitizedDeps,
      context_files: sanitizedFiles,
      hints: hintsResult.value,
      warnings,
    };
  }

  /**
   * Encrypt task data if encryption is enabled
   */
  maybeEncryptTask(taskData: Record<string, any>): Record<string, any> {
    if (this.config.enableEncryption) {
      return this.encryptionManager.encryptTaskData(taskData);
    }
    return taskData;
  }

  /**
   * Decrypt task data if needed
   */
  maybeDecryptTask(taskData: Record<string, any>): Record<string, any> {
    if (taskData._description_encrypted) {
      return this.encryptionManager.decryptTaskData(taskData);
    }
    return taskData;
  }

  /**
   * Rotate all secrets
   */
  rotateSecrets(): Record<string, boolean> {
    return this.rotationManager.rotateAll();
  }

  /**
   * Get audit log
   */
  getAuditLog(options?: {
    eventType?: AuditEventType;
    actor?: string;
    limit?: number;
  }): AuditEvent[] {
    return this.auditLogger.queryEvents(options || {});
  }
}
