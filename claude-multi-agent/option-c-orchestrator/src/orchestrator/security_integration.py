"""
Security Integration for Option C (Orchestrator).

This module wraps the Orchestrator with comprehensive security features
from the shared security module.

Implements:
- adv-sec-001: API key authentication for agents
- adv-sec-002: JWT token support with expiration
- adv-sec-003: Role-based access control (leader/worker/admin)
- adv-sec-004: Task data encryption at rest
- adv-sec-005: Secure credential storage
- adv-sec-006: Audit logging for all operations
- adv-sec-007: Input sanitization for task descriptions
- adv-sec-008: Rate limiting per identity
- adv-sec-009: mTLS support for inter-agent communication
- adv-sec-010: Secret rotation mechanism
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Any, Dict

# Add shared security module to path
SHARED_SECURITY_PATH = Path(__file__).parent.parent.parent.parent.parent / "shared" / "security"
if str(SHARED_SECURITY_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SHARED_SECURITY_PATH.parent))

from security import (  # noqa: E402
    # Authentication (sec-001, sec-002, sec-003)
    APIKeyManager,
    JWTManager,
    RBACManager,
    Role,
    Permission,
    # Encryption (sec-004, sec-005)
    EncryptionManager,
    SecureCredentialStore,
    # Audit (sec-006)
    AuditLogger,
    AuditEventType,
    # Sanitization (sec-007)
    InputSanitizer,
    SanitizationError,
    # Rate Limiting (sec-008)
    RateLimiter,
    RateLimitExceeded,
    # mTLS (sec-009)
    MTLSManager,
    # Rotation (sec-010)
    SecretRotationManager,
    RotationSchedule,
)

from .async_orchestrator import Orchestrator as AsyncOrchestrator  # noqa: E402
from .models import (  # noqa: E402
    Task,
    TaskResult,
    Discovery,
    CoordinationState,
)
from .config import DEFAULT_MODEL, DEFAULT_MAX_WORKERS, DEFAULT_TASK_TIMEOUT  # noqa: E402


class SecurityConfig:
    """Configuration for security features."""

    def __init__(
        self,
        enable_auth: bool = True,
        enable_jwt: bool = True,
        enable_rbac: bool = True,
        enable_encryption: bool = True,
        enable_audit: bool = True,
        enable_sanitization: bool = True,
        enable_rate_limiting: bool = True,
        enable_mtls: bool = False,
        enable_rotation: bool = True,
        encryption_key: Optional[bytes] = None,
        jwt_secret: Optional[str] = None,
        audit_log_path: Optional[str] = None,
        cert_dir: Optional[str] = None,
        rate_limit_requests: int = 100,
        rate_limit_window: int = 60,
    ):
        self.enable_auth = enable_auth
        self.enable_jwt = enable_jwt
        self.enable_rbac = enable_rbac
        self.enable_encryption = enable_encryption
        self.enable_audit = enable_audit
        self.enable_sanitization = enable_sanitization
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_mtls = enable_mtls
        self.enable_rotation = enable_rotation
        self.encryption_key = encryption_key
        self.jwt_secret = jwt_secret
        self.audit_log_path = audit_log_path
        self.cert_dir = cert_dir
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window = rate_limit_window


class SecureOrchestrator:
    """
    A security-enhanced wrapper around the Orchestrator.

    Provides:
    - Authentication and authorization for all agent operations
    - Encrypted task storage
    - Comprehensive audit logging
    - Input validation and sanitization
    - Rate limiting for API operations
    - mTLS for secure inter-agent communication
    - Automatic secret rotation
    """

    def __init__(
        self,
        working_directory: str = ".",
        max_workers: int = DEFAULT_MAX_WORKERS,
        model: str = DEFAULT_MODEL,
        task_timeout: int = DEFAULT_TASK_TIMEOUT,
        security_config: Optional[SecurityConfig] = None,
        on_task_complete: Optional[Callable[[Task], None]] = None,
        on_discovery: Optional[Callable[[Discovery], None]] = None,
        verbose: bool = True,
    ):
        """
        Initialize the secure orchestrator.

        Args:
            working_directory: Base directory for coordination
            max_workers: Maximum number of worker agents
            model: Claude model to use
            task_timeout: Task timeout in seconds
            security_config: Security configuration
            on_task_complete: Callback for task completion
            on_discovery: Callback for new discoveries
            verbose: Enable verbose output
        """
        self.config = security_config or SecurityConfig()

        # Initialize underlying orchestrator
        self._orchestrator = AsyncOrchestrator(
            working_directory=working_directory,
            max_workers=max_workers,
            model=model,
            task_timeout=task_timeout,
            on_task_complete=on_task_complete,
            on_discovery=on_discovery,
            verbose=verbose,
        )

        # Initialize security components
        self._init_security_components()

        # Track authenticated sessions
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def _init_security_components(self) -> None:
        """Initialize all security components based on config."""
        # sec-001: API Key Authentication
        if self.config.enable_auth:
            self._api_key_manager = APIKeyManager()
        else:
            self._api_key_manager = None

        # sec-002: JWT Token Support
        if self.config.enable_jwt:
            self._jwt_manager = JWTManager(
                secret_key=self.config.jwt_secret or "orchestrator-jwt-secret"
            )
        else:
            self._jwt_manager = None

        # sec-003: Role-Based Access Control
        if self.config.enable_rbac:
            self._rbac_manager = RBACManager()
        else:
            self._rbac_manager = None

        # sec-004: Task Encryption
        if self.config.enable_encryption:
            self._encryption_manager = EncryptionManager(
                key=self.config.encryption_key
            )
        else:
            self._encryption_manager = None

        # sec-005: Secure Credential Storage
        if self.config.enable_encryption:
            credential_path = Path(self._orchestrator.working_directory) / ".credentials"
            self._credential_store = SecureCredentialStore(
                storage_path=str(credential_path)
            )
        else:
            self._credential_store = None

        # sec-006: Audit Logging
        if self.config.enable_audit:
            audit_path = self.config.audit_log_path or str(
                Path(self._orchestrator.working_directory) / ".coordination" / "audit.log"
            )
            self._audit_logger = AuditLogger(log_path=audit_path)
        else:
            self._audit_logger = None

        # sec-007: Input Sanitization
        if self.config.enable_sanitization:
            self._sanitizer = InputSanitizer()
        else:
            self._sanitizer = None

        # sec-008: Rate Limiting
        if self.config.enable_rate_limiting:
            self._rate_limiter = RateLimiter(
                max_requests=self.config.rate_limit_requests,
                window_seconds=self.config.rate_limit_window,
            )
        else:
            self._rate_limiter = None

        # sec-009: mTLS Support
        if self.config.enable_mtls:
            cert_dir = self.config.cert_dir or str(
                Path(self._orchestrator.working_directory) / ".certs"
            )
            self._mtls_manager = MTLSManager(cert_dir=cert_dir)
        else:
            self._mtls_manager = None

        # sec-010: Secret Rotation
        if self.config.enable_rotation:
            self._rotation_manager = SecretRotationManager()
        else:
            self._rotation_manager = None

    # =========================================================================
    # Authentication Methods (sec-001, sec-002)
    # =========================================================================

    def register_agent(
        self,
        agent_id: str,
        role: Role = Role.WORKER,
    ) -> Dict[str, str]:
        """
        Register a new agent with credentials.

        Returns:
            Dictionary containing api_key and jwt_token
        """
        credentials = {}

        if self._api_key_manager:
            api_key = self._api_key_manager.create_key(agent_id)
            credentials["api_key"] = api_key

        if self._jwt_manager:
            token = self._jwt_manager.create_token(
                subject=agent_id,
                claims={"role": role.value},
            )
            credentials["jwt_token"] = token

        if self._rbac_manager:
            self._rbac_manager.assign_role(agent_id, role)

        if self._audit_logger:
            self._audit_logger.log(
                event_type=AuditEventType.AUTH,
                agent_id=agent_id,
                action="register",
                details={"role": role.value},
            )

        self._sessions[agent_id] = {
            "role": role,
            "authenticated_at": datetime.now(),
        }

        return credentials

    def authenticate_agent(
        self,
        agent_id: str,
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
    ) -> bool:
        """
        Authenticate an agent using API key or JWT token.

        Args:
            agent_id: The agent's identifier
            api_key: Optional API key
            jwt_token: Optional JWT token

        Returns:
            True if authentication succeeds
        """
        # Check rate limiting first
        if self._rate_limiter:
            try:
                self._rate_limiter.check(agent_id)
            except RateLimitExceeded:
                if self._audit_logger:
                    self._audit_logger.log(
                        event_type=AuditEventType.SECURITY,
                        agent_id=agent_id,
                        action="rate_limit_exceeded",
                        details={},
                    )
                return False

        authenticated = False

        # Try API key authentication
        if api_key and self._api_key_manager:
            authenticated = self._api_key_manager.validate_key(agent_id, api_key)

        # Try JWT authentication
        if not authenticated and jwt_token and self._jwt_manager:
            payload = self._jwt_manager.verify_token(jwt_token)
            if payload and payload.get("sub") == agent_id:
                authenticated = True

        # Log authentication attempt
        if self._audit_logger:
            self._audit_logger.log(
                event_type=AuditEventType.AUTH,
                agent_id=agent_id,
                action="authenticate",
                details={"success": authenticated},
            )

        if authenticated:
            self._sessions[agent_id] = {
                "authenticated_at": datetime.now(),
            }

        return authenticated

    def _check_permission(
        self,
        agent_id: str,
        permission: Permission,
    ) -> bool:
        """Check if agent has a specific permission."""
        if not self._rbac_manager:
            return True  # No RBAC enabled, allow all

        return self._rbac_manager.has_permission(agent_id, permission)

    # =========================================================================
    # Orchestrator Wrapper Methods
    # =========================================================================

    async def initialize(
        self,
        goal: str,
        master_plan: str = "",
        agent_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the orchestration session with security checks.

        Args:
            goal: The overall project goal
            master_plan: Optional high-level plan
            agent_id: ID of the agent making the request (for audit)
        """
        # Sanitize input
        if self._sanitizer:
            try:
                goal = self._sanitizer.sanitize(goal)
                if master_plan:
                    master_plan = self._sanitizer.sanitize(master_plan)
            except SanitizationError as e:
                if self._audit_logger:
                    self._audit_logger.log(
                        event_type=AuditEventType.SECURITY,
                        agent_id=agent_id or "system",
                        action="sanitization_failure",
                        details={"error": str(e)},
                    )
                raise

        await self._orchestrator.initialize(goal, master_plan)

        if self._audit_logger:
            self._audit_logger.log(
                event_type=AuditEventType.TASK,
                agent_id=agent_id or "system",
                action="session_initialized",
                details={"goal": goal[:100]},
            )

    async def start(self, agent_id: Optional[str] = None) -> None:
        """Start the orchestrator with security initialization."""
        # Initialize mTLS certificates if enabled
        if self._mtls_manager:
            self._mtls_manager.generate_ca()

        await self._orchestrator.start()

        if self._audit_logger:
            self._audit_logger.log(
                event_type=AuditEventType.SYSTEM,
                agent_id=agent_id or "system",
                action="orchestrator_started",
                details={"max_workers": self._orchestrator.max_workers},
            )

    async def stop(self, agent_id: Optional[str] = None) -> None:
        """Stop the orchestrator."""
        await self._orchestrator.stop()

        if self._audit_logger:
            self._audit_logger.log(
                event_type=AuditEventType.SYSTEM,
                agent_id=agent_id or "system",
                action="orchestrator_stopped",
                details={},
            )

    def add_task(
        self,
        description: str,
        priority: int = 5,
        dependencies: Optional[list[str]] = None,
        context_files: Optional[list[str]] = None,
        hints: str = "",
        agent_id: Optional[str] = None,
        encrypt: bool = True,
    ) -> Task:
        """
        Add a new task with security features.

        Args:
            description: Task description
            priority: Priority level (1-10)
            dependencies: List of dependency task IDs
            context_files: Relevant file paths
            hints: Additional hints
            agent_id: ID of the requesting agent
            encrypt: Whether to encrypt task data

        Returns:
            The created task
        """
        # Check permission
        if agent_id and not self._check_permission(agent_id, Permission.TASK_CREATE):
            raise PermissionError(f"Agent {agent_id} lacks permission to create tasks")

        # Rate limiting
        if agent_id and self._rate_limiter:
            self._rate_limiter.check(agent_id)

        # Sanitize input
        if self._sanitizer:
            description = self._sanitizer.sanitize(description)
            if hints:
                hints = self._sanitizer.sanitize(hints)

        # Encrypt task description if enabled
        encrypted_description = description
        if encrypt and self._encryption_manager:
            encrypted_description = self._encryption_manager.encrypt(description)

        task = self._orchestrator.add_task(
            description=encrypted_description,
            priority=priority,
            dependencies=dependencies,
            context_files=context_files,
            hints=hints,
        )

        # Store encryption status in metadata
        if encrypt and self._encryption_manager:
            task.context.metadata["encrypted"] = True

        # Audit log
        if self._audit_logger:
            self._audit_logger.log(
                event_type=AuditEventType.TASK,
                agent_id=agent_id or "system",
                action="task_created",
                resource_type="task",
                resource_id=task.id,
                details={"priority": priority, "encrypted": encrypt},
            )

        return task

    async def claim_task(
        self,
        agent_id: str,
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
    ) -> Optional[Task]:
        """
        Claim an available task with authentication.

        Args:
            agent_id: The claiming agent's ID
            api_key: Optional API key for authentication
            jwt_token: Optional JWT token for authentication

        Returns:
            The claimed task, or None if none available
        """
        # Authenticate
        if (api_key or jwt_token) and not self.authenticate_agent(
            agent_id, api_key, jwt_token
        ):
            raise PermissionError(f"Authentication failed for agent {agent_id}")

        # Check permission
        if not self._check_permission(agent_id, Permission.TASK_CLAIM):
            raise PermissionError(f"Agent {agent_id} lacks permission to claim tasks")

        task = await self._orchestrator.claim_task(agent_id)

        if task and self._audit_logger:
            self._audit_logger.log(
                event_type=AuditEventType.TASK,
                agent_id=agent_id,
                action="task_claimed",
                resource_type="task",
                resource_id=task.id,
                details={},
            )

        return task

    async def complete_task(
        self,
        task_id: str,
        result: TaskResult,
        agent_id: Optional[str] = None,
    ) -> bool:
        """
        Mark a task as completed.

        Args:
            task_id: The task ID
            result: The task result
            agent_id: ID of the completing agent

        Returns:
            True if successful
        """
        # Check permission
        if agent_id and not self._check_permission(agent_id, Permission.TASK_COMPLETE):
            raise PermissionError(f"Agent {agent_id} lacks permission to complete tasks")

        success = await self._orchestrator.complete_task(task_id, result)

        if self._audit_logger:
            self._audit_logger.log(
                event_type=AuditEventType.TASK,
                agent_id=agent_id or "unknown",
                action="task_completed" if success else "task_complete_failed",
                resource_type="task",
                resource_id=task_id,
                details={"success": success},
            )

        return success

    async def fail_task(
        self,
        task_id: str,
        error: str,
        agent_id: Optional[str] = None,
    ) -> bool:
        """
        Mark a task as failed.

        Args:
            task_id: The task ID
            error: Error description
            agent_id: ID of the reporting agent

        Returns:
            True if successful
        """
        success = await self._orchestrator.fail_task(task_id, error)

        if self._audit_logger:
            self._audit_logger.log(
                event_type=AuditEventType.TASK,
                agent_id=agent_id or "unknown",
                action="task_failed",
                resource_type="task",
                resource_id=task_id,
                details={"error": error[:200]},
            )

        return success

    # =========================================================================
    # mTLS Certificate Management (sec-009)
    # =========================================================================

    def generate_agent_certificate(
        self,
        agent_id: str,
        admin_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate mTLS certificate for an agent.

        Args:
            agent_id: The agent's identifier
            admin_id: ID of the admin requesting (for audit)

        Returns:
            Dictionary with cert_path and key_path
        """
        if not self._mtls_manager:
            raise RuntimeError("mTLS is not enabled")

        # Check admin permission
        if admin_id and not self._check_permission(admin_id, Permission.ADMIN):
            raise PermissionError("Only admins can generate certificates")

        cert_path, key_path = self._mtls_manager.generate_agent_cert(agent_id)

        if self._audit_logger:
            self._audit_logger.log(
                event_type=AuditEventType.SECURITY,
                agent_id=admin_id or "system",
                action="certificate_generated",
                details={"target_agent": agent_id},
            )

        return {
            "cert_path": cert_path,
            "key_path": key_path,
        }

    # =========================================================================
    # Secret Rotation (sec-010)
    # =========================================================================

    def rotate_secrets(
        self,
        admin_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Rotate all secrets (API keys, JWT secret, encryption key).

        Args:
            admin_id: ID of the admin initiating rotation

        Returns:
            Dictionary with rotation results
        """
        results = {}

        # Check admin permission
        if admin_id and not self._check_permission(admin_id, Permission.ADMIN):
            raise PermissionError("Only admins can rotate secrets")

        if self._rotation_manager:
            # Schedule rotations
            if self._api_key_manager:
                self._rotation_manager.schedule_rotation(
                    secret_id="api_keys",
                    rotation_function=self._rotate_api_keys,
                    schedule=RotationSchedule.DAILY,
                )

            if self._jwt_manager:
                self._rotation_manager.schedule_rotation(
                    secret_id="jwt_secret",
                    rotation_function=self._rotate_jwt_secret,
                    schedule=RotationSchedule.WEEKLY,
                )

            if self._encryption_manager:
                self._rotation_manager.schedule_rotation(
                    secret_id="encryption_key",
                    rotation_function=self._rotate_encryption_key,
                    schedule=RotationSchedule.MONTHLY,
                )

            # Execute pending rotations
            rotated = self._rotation_manager.execute_pending_rotations()
            results["rotated"] = rotated

        if self._audit_logger:
            self._audit_logger.log(
                event_type=AuditEventType.SECURITY,
                agent_id=admin_id or "system",
                action="secrets_rotated",
                details=results,
            )

        return results

    def _rotate_api_keys(self) -> bool:
        """Rotate all API keys."""
        if not self._api_key_manager:
            return False

        # Regenerate keys for all registered agents
        for agent_id in list(self._sessions.keys()):
            self._api_key_manager.create_key(agent_id)
            self._sessions[agent_id]["api_key_rotated"] = datetime.now()

        return True

    def _rotate_jwt_secret(self) -> bool:
        """Rotate JWT signing secret."""
        if not self._jwt_manager:
            return False

        # Generate new secret and reinitialize
        import secrets
        new_secret = secrets.token_hex(32)
        self._jwt_manager = JWTManager(secret_key=new_secret)
        return True

    def _rotate_encryption_key(self) -> bool:
        """Rotate encryption key."""
        if not self._encryption_manager:
            return False

        # Generate new key (requires re-encryption of existing data)
        new_key = EncryptionManager.generate_key()
        self._encryption_manager = EncryptionManager(key=new_key)
        return True

    # =========================================================================
    # Secure Credential Storage (sec-005)
    # =========================================================================

    def store_credential(
        self,
        name: str,
        value: str,
        agent_id: Optional[str] = None,
    ) -> None:
        """
        Securely store a credential.

        Args:
            name: Credential name/key
            value: Credential value
            agent_id: ID of the storing agent
        """
        if not self._credential_store:
            raise RuntimeError("Credential storage is not enabled")

        # Check permission
        if agent_id and not self._check_permission(agent_id, Permission.ADMIN):
            raise PermissionError("Only admins can store credentials")

        self._credential_store.store(name, value)

        if self._audit_logger:
            self._audit_logger.log(
                event_type=AuditEventType.SECURITY,
                agent_id=agent_id or "system",
                action="credential_stored",
                details={"name": name},
            )

    def retrieve_credential(
        self,
        name: str,
        agent_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Retrieve a stored credential.

        Args:
            name: Credential name/key
            agent_id: ID of the retrieving agent

        Returns:
            The credential value or None
        """
        if not self._credential_store:
            raise RuntimeError("Credential storage is not enabled")

        # Check permission
        if agent_id and not self._check_permission(agent_id, Permission.ADMIN):
            raise PermissionError("Only admins can retrieve credentials")

        value = self._credential_store.retrieve(name)

        if self._audit_logger:
            self._audit_logger.log(
                event_type=AuditEventType.SECURITY,
                agent_id=agent_id or "system",
                action="credential_retrieved",
                details={"name": name, "found": value is not None},
            )

        return value

    # =========================================================================
    # Delegation to Underlying Orchestrator
    # =========================================================================

    async def run_with_leader_planning(
        self,
        admin_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full orchestration flow with security."""
        if admin_id and not self._check_permission(admin_id, Permission.ADMIN):
            raise PermissionError("Only admins can run full orchestration")

        if self._audit_logger:
            self._audit_logger.log(
                event_type=AuditEventType.SYSTEM,
                agent_id=admin_id or "system",
                action="orchestration_started",
                details={"mode": "leader_planning"},
            )

        result = await self._orchestrator.run_with_leader_planning()

        if self._audit_logger:
            self._audit_logger.log(
                event_type=AuditEventType.SYSTEM,
                agent_id=admin_id or "system",
                action="orchestration_completed",
                details={
                    "tasks_completed": result.get("tasks_completed", 0),
                    "tasks_failed": result.get("tasks_failed", 0),
                },
            )

        return result

    async def run_with_predefined_tasks(
        self,
        admin_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run orchestration with predefined tasks."""
        if admin_id and not self._check_permission(admin_id, Permission.ADMIN):
            raise PermissionError("Only admins can run orchestration")

        if self._audit_logger:
            self._audit_logger.log(
                event_type=AuditEventType.SYSTEM,
                agent_id=admin_id or "system",
                action="orchestration_started",
                details={"mode": "predefined_tasks"},
            )

        result = await self._orchestrator.run_with_predefined_tasks()

        if self._audit_logger:
            self._audit_logger.log(
                event_type=AuditEventType.SYSTEM,
                agent_id=admin_id or "system",
                action="orchestration_completed",
                details={
                    "tasks_completed": result.get("tasks_completed", 0),
                    "tasks_failed": result.get("tasks_failed", 0),
                },
            )

        return result

    def get_status(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get orchestrator status with security info."""
        status = self._orchestrator.get_status()

        # Add security info
        status["security"] = {
            "auth_enabled": self._api_key_manager is not None,
            "jwt_enabled": self._jwt_manager is not None,
            "rbac_enabled": self._rbac_manager is not None,
            "encryption_enabled": self._encryption_manager is not None,
            "audit_enabled": self._audit_logger is not None,
            "mtls_enabled": self._mtls_manager is not None,
            "active_sessions": len(self._sessions),
        }

        return status

    def save_state(self, filepath: str, encrypt: bool = True) -> None:
        """Save state with optional encryption."""
        if encrypt and self._encryption_manager:
            # Save encrypted state
            import json
            state_json = json.dumps(
                self._orchestrator.state.model_dump(mode="json"),
                indent=2,
                default=str,
            )
            encrypted = self._encryption_manager.encrypt(state_json)
            with open(filepath + ".enc", "w") as f:
                f.write(encrypted)
        else:
            self._orchestrator.save_state(filepath)

    def load_state(self, filepath: str, encrypted: bool = False) -> None:
        """Load state with optional decryption."""
        if encrypted and self._encryption_manager:
            import json
            with open(filepath, "r") as f:
                encrypted_data = f.read()
            decrypted = self._encryption_manager.decrypt(encrypted_data)
            data = json.loads(decrypted)
            self._orchestrator.state = CoordinationState.model_validate(data)
        else:
            self._orchestrator.load_state(filepath)

    # =========================================================================
    # Audit Log Access
    # =========================================================================

    def get_audit_log(
        self,
        admin_id: str,
        limit: int = 100,
        event_type: Optional[AuditEventType] = None,
    ) -> list[Dict[str, Any]]:
        """
        Retrieve audit log entries.

        Args:
            admin_id: Admin requesting the logs
            limit: Maximum entries to return
            event_type: Optional filter by event type

        Returns:
            List of audit log entries
        """
        if not self._check_permission(admin_id, Permission.ADMIN):
            raise PermissionError("Only admins can access audit logs")

        if not self._audit_logger:
            return []

        return self._audit_logger.get_entries(
            limit=limit,
            event_type=event_type,
        )


# Convenience function for creating a secure orchestrator
def create_secure_orchestrator(
    working_directory: str = ".",
    max_workers: int = 3,
    enable_all_security: bool = True,
    **kwargs,
) -> SecureOrchestrator:
    """
    Factory function to create a SecureOrchestrator with sensible defaults.

    Args:
        working_directory: Base directory for coordination
        max_workers: Maximum worker agents
        enable_all_security: Enable all security features
        **kwargs: Additional arguments passed to SecureOrchestrator

    Returns:
        Configured SecureOrchestrator instance
    """
    config = SecurityConfig(
        enable_auth=enable_all_security,
        enable_jwt=enable_all_security,
        enable_rbac=enable_all_security,
        enable_encryption=enable_all_security,
        enable_audit=enable_all_security,
        enable_sanitization=enable_all_security,
        enable_rate_limiting=enable_all_security,
        enable_mtls=False,  # mTLS requires explicit setup
        enable_rotation=enable_all_security,
    )

    return SecureOrchestrator(
        working_directory=working_directory,
        max_workers=max_workers,
        security_config=config,
        **kwargs,
    )
