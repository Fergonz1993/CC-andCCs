/**
 * Shared type definitions for the Claude Multi-Agent Coordination MCP Server
 */

// ============================================================================
// Core Task Types
// ============================================================================

export type TaskStatus =
  | "available"
  | "claimed"
  | "in_progress"
  | "done"
  | "failed";

export interface Task {
  id: string;
  description: string;
  status: TaskStatus;
  priority: number;
  claimed_by: string | null;
  dependencies: string[];
  context: {
    files?: string[];
    hints?: string;
    parent_task?: string;
  } | null;
  result: {
    output?: string;
    files_modified?: string[];
    files_created?: string[];
    error?: string;
  } | null;
  created_at: string;
  claimed_at: string | null;
  completed_at: string | null;
  tags?: string[];
}

export interface Agent {
  id: string;
  role: "leader" | "worker";
  last_heartbeat: string;
  current_task: string | null;
  tasks_completed: number;
}

export interface Discovery {
  id: string;
  agent_id: string;
  content: string;
  tags: string[];
  created_at: string;
}

// ============================================================================
// Audit Log Types (adv-b-009)
// ============================================================================

export interface AuditLogEntry {
  id: string;
  timestamp: string;
  action: string;
  entity_type: "task" | "agent" | "discovery" | "coordination" | "transaction";
  entity_id: string;
  agent_id: string | null;
  details?: Record<string, unknown>;
}

// ============================================================================
// Rate Limiting Types (adv-b-003)
// ============================================================================

export interface RateLimitConfig {
  max_requests: number;
  window_ms: number;
  tokens: number;
  last_refill: number;
}

export interface RateLimitBucket {
  tokens: number;
  last_refill: number;
}

// ============================================================================
// Request Queue Types (adv-b-004)
// ============================================================================

export interface QueuedRequest {
  id: string;
  priority: number;
  timestamp: number;
  agent_id: string;
  tool_name: string;
  args: Record<string, unknown>;
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
}

// ============================================================================
// Transaction Types (adv-b-006)
// ============================================================================

export interface Transaction {
  id: string;
  agent_id: string;
  operations: TransactionOperation[];
  status: "pending" | "committed" | "rolled_back" | "failed";
  created_at: string;
  committed_at?: string;
}

export interface TransactionOperation {
  type: string;
  params: Record<string, unknown>;
}

// ============================================================================
// WebSocket Types (adv-b-001)
// ============================================================================

export interface WebSocketClient {
  id: string;
  agent_id: string | null;
  subscriptions: Set<string>;
  send: (message: string) => void;
}

export type WebSocketEventType =
  | "task_update"
  | "agent_update"
  | "discovery_added"
  | "coordination_status"
  | "heartbeat";

export interface WebSocketMessage {
  type: WebSocketEventType;
  payload: unknown;
  timestamp: string;
}

// ============================================================================
// Health Check Types (adv-b-010)
// ============================================================================

export interface HealthStatus {
  status: "healthy" | "degraded" | "unhealthy";
  uptime_seconds: number;
  version: string;
  checks: {
    storage: HealthCheck;
    agents: HealthCheck;
    tasks: HealthCheck;
    memory: HealthCheck;
  };
  timestamp: string;
}

export interface HealthCheck {
  status: "pass" | "fail" | "warn";
  message: string;
  details?: Record<string, unknown>;
}

// ============================================================================
// Query/Filter Types (adv-b-007)
// ============================================================================

export interface TaskFilter {
  status?: TaskStatus | TaskStatus[];
  priority_min?: number;
  priority_max?: number;
  claimed_by?: string;
  tags?: string[];
  created_after?: string;
  created_before?: string;
  has_dependencies?: boolean;
  search?: string;
}

// ============================================================================
// Pagination Types (adv-b-008)
// ============================================================================

export interface PaginationOptions {
  page?: number;
  page_size?: number;
  sort_by?: string;
  sort_direction?: "asc" | "desc";
}

export interface PaginatedResult<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}

// ============================================================================
// Batch Operation Types (adv-b-005)
// ============================================================================

export interface BatchOperation {
  operation:
    | "create_task"
    | "update_task"
    | "delete_task"
    | "claim_task"
    | "complete_task";
  params: Record<string, unknown>;
}

export interface BatchResult {
  operation_index: number;
  success: boolean;
  data?: unknown;
  error?: string;
}

// ============================================================================
// State Management
// ============================================================================

export interface CoordinationState {
  master_plan: string;
  goal: string;
  tasks: Task[];
  agents: Map<string, Agent>;
  discoveries: Discovery[];
  created_at: string;
  last_activity: string;
  audit_log: AuditLogEntry[];
  transactions: Map<string, Transaction>;
}

export interface SerializableState {
  master_plan: string;
  goal: string;
  tasks: Task[];
  agents: Record<string, Agent>;
  discoveries: Discovery[];
  created_at: string;
  last_activity: string;
  audit_log: AuditLogEntry[];
  transactions: Record<string, Transaction>;
}

// ============================================================================
// Subscription Types (adv-b-sub-001 through adv-b-sub-005)
// ============================================================================

export type SubscriptionEventType =
  | "task_created"
  | "task_updated"
  | "task_claimed"
  | "task_started"
  | "task_completed"
  | "task_failed"
  | "discovery_added"
  | "agent_registered"
  | "agent_heartbeat"
  | "coordination_init";

export interface SubscriptionFilter {
  event_types?: SubscriptionEventType[];
  task_status?: TaskStatus[];
  task_tags?: string[];
  agent_id?: string;
  priority_min?: number;
  priority_max?: number;
}

export interface Subscription {
  id: string;
  subscriber_id: string;
  filter: SubscriptionFilter;
  created_at: string;
}

export interface Notification {
  id: string;
  subscription_id: string;
  event_type: SubscriptionEventType;
  payload: unknown;
  timestamp: string;
}

export interface BatchedNotifications {
  batch_id: string;
  subscriber_id: string;
  notifications: Notification[];
  start_time: string;
  end_time: string;
}

// ============================================================================
// Webhook Types (adv-b-sub-002)
// ============================================================================

export interface WebhookConfig {
  id: string;
  url: string;
  secret: string;
  events: SubscriptionEventType[];
  enabled: boolean;
  created_at: string;
  headers?: Record<string, string>;
  retry_config: WebhookRetryConfig;
  filter?: WebhookFilter;
}

export interface WebhookFilter {
  task_status?: TaskStatus[];
  task_tags?: string[];
  agent_id?: string;
  priority_min?: number;
  priority_max?: number;
}

export interface WebhookRetryConfig {
  max_retries: number;
  initial_delay_ms: number;
  max_delay_ms: number;
  backoff_multiplier: number;
}

export interface WebhookDelivery {
  id: string;
  webhook_id: string;
  event_type: SubscriptionEventType;
  payload: unknown;
  status: "pending" | "success" | "failed" | "retrying";
  attempts: number;
  created_at: string;
  last_attempt_at: string | null;
  next_retry_at: string | null;
  response_status?: number;
  response_body?: string;
  error?: string;
}

// ============================================================================
// SSE Types (adv-b-sub-003)
// ============================================================================

export interface SSEClient {
  id: string;
  agent_id: string | null;
  connected_at: string;
  last_event_at: string | null;
  filter: SSEFilter;
}

export interface SSEFilter {
  event_types?: SubscriptionEventType[];
  task_status?: TaskStatus[];
  task_tags?: string[];
  agent_id?: string;
  priority_min?: number;
  priority_max?: number;
}

export interface SSEEvent {
  id: string;
  event: SubscriptionEventType;
  data: unknown;
  retry?: number;
}

// ============================================================================
// Security Types (adv-b-sec-001 through adv-b-sec-005)
// ============================================================================

export interface APIKey {
  id: string;
  key: string;
  name: string;
  agent_id: string | null;
  role: "leader" | "worker" | "admin";
  permissions: Permission[];
  rate_limit: APIKeyRateLimit;
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

export interface APIKeyRateLimit {
  requests_per_minute: number;
  requests_per_hour: number;
  burst_limit: number;
}

export interface JWTPayload {
  sub: string;
  iss: string;
  aud: string;
  exp: number;
  iat: number;
  jti: string;
  role: "leader" | "worker" | "admin";
  permissions: Permission[];
}

export interface RequestSignature {
  algorithm: "sha256" | "sha512";
  timestamp: number;
  nonce: string;
  signature: string;
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
