#!/usr/bin/env python3
"""
Security Integration for Option A (File-Based) Coordination System.

This module integrates the shared security features with the file-based
coordination system, providing:
- API key authentication
- JWT token support
- Role-based access control
- Task data encryption
- Secure credential storage
- Audit logging
- Input sanitization
- Rate limiting
- mTLS support
- Secret rotation

Usage:
    from security_integration import SecureCoordinator

    # Initialize with security enabled
    coordinator = SecureCoordinator(admin_key="your-admin-key")

    # Authenticate and get a token
    token = coordinator.authenticate("mca_xxx.yyy")

    # Use token for operations
    coordinator.leader_add_task(token, "Task description", priority=1)
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from functools import wraps

# Add shared security module to path
shared_path = Path(__file__).parent.parent / "shared" / "security"
sys.path.insert(0, str(shared_path.parent.parent))

from shared.security.auth import (
    APIKeyManager,
    JWTManager,
    RBACManager,
    Role,
    Permission,
)
from shared.security.encryption import EncryptionManager, SecureCredentialStore
from shared.security.audit import AuditLogger, AuditEventType
from shared.security.sanitization import InputSanitizer, SanitizationMode, sanitize_task_input
from shared.security.rate_limiting import (
    RateLimiter,
    RateLimitConfig,
    RateLimitAlgorithm,
    RateLimitExceeded,
    create_default_rate_limits,
)
from shared.security.mtls import MTLSManager
from shared.security.rotation import SecretRotationManager, SecretType, RotationSchedule

# Import the original coordination module
from coordination import (
    Task,
    COORDINATION_DIR,
    TASKS_FILE,
    ensure_coordination_structure,
    load_tasks,
    save_tasks,
    now_iso,
    generate_task_id,
    log_action,
)


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class AuthorizationError(Exception):
    """Raised when authorization fails."""
    pass


class SecureCoordinator:
    """
    Secure wrapper around the file-based coordination system.

    Provides all the original coordination functionality with added security.
    """

    def __init__(
        self,
        coordination_dir: Optional[str] = None,
        admin_key: Optional[str] = None,
        encryption_key: Optional[str] = None,
        enable_encryption: bool = True,
        enable_audit: bool = True,
        enable_rate_limiting: bool = True,
        mtls_enabled: bool = False,
    ):
        """
        Initialize the secure coordinator.

        Args:
            coordination_dir: Override for coordination directory
            admin_key: Initial admin API key (generated if not provided)
            encryption_key: Master encryption key (generated if not provided)
            enable_encryption: Whether to encrypt task data at rest
            enable_audit: Whether to enable audit logging
            enable_rate_limiting: Whether to enable rate limiting
            mtls_enabled: Whether to enable mTLS for agent communication
        """
        self.coord_dir = Path(coordination_dir) if coordination_dir else COORDINATION_DIR
        self.security_dir = self.coord_dir / "security"
        self.security_dir.mkdir(parents=True, exist_ok=True)

        # Initialize security components
        self._init_api_keys(admin_key)
        self._init_jwt()
        self._init_rbac()
        self._init_encryption(encryption_key, enable_encryption)
        self._init_audit(enable_audit)
        self._init_rate_limiting(enable_rate_limiting)
        self._init_secret_rotation()

        if mtls_enabled:
            self._init_mtls()
        else:
            self.mtls_manager = None

        # Input sanitizer
        self.sanitizer = InputSanitizer(mode=SanitizationMode.MODERATE)

        # Ensure coordination structure exists
        ensure_coordination_structure()

    def _init_api_keys(self, admin_key: Optional[str]) -> None:
        """Initialize API key management."""
        self.api_key_manager = APIKeyManager(
            storage_path=str(self.security_dir / "api_keys.json")
        )

        # Generate admin key if not provided and no keys exist
        if not self.api_key_manager.list_keys():
            if admin_key:
                # Use provided key as initial admin
                self._initial_admin_key = admin_key
            else:
                # Generate new admin key
                key_id, key = self.api_key_manager.generate_key(
                    name="initial_admin",
                    role=Role.ADMIN,
                )
                self._initial_admin_key = key
                print(f"Generated initial admin API key: {key}")
                print("Please save this key securely - it cannot be retrieved later.")

    def _init_jwt(self) -> None:
        """Initialize JWT management."""
        jwt_secret = os.environ.get('JWT_SECRET_KEY')
        self.jwt_manager = JWTManager(
            secret_key=jwt_secret,
            access_token_expire_minutes=60,
            refresh_token_expire_days=7,
        )

    def _init_rbac(self) -> None:
        """Initialize RBAC."""
        self.rbac_manager = RBACManager()

    def _init_encryption(self, key: Optional[str], enabled: bool) -> None:
        """Initialize encryption."""
        self.encryption_enabled = enabled
        if enabled:
            self.encryption_manager = EncryptionManager(master_key=key)
            self.credential_store = SecureCredentialStore(
                storage_path=str(self.security_dir / "credentials.enc"),
                encryption_manager=self.encryption_manager,
            )
        else:
            self.encryption_manager = None
            self.credential_store = None

    def _init_audit(self, enabled: bool) -> None:
        """Initialize audit logging."""
        self.audit_enabled = enabled
        if enabled:
            self.audit_logger = AuditLogger(
                log_file=str(self.security_dir / "audit.log"),
                enable_console=False,
                enable_tamper_detection=True,
            )
        else:
            self.audit_logger = None

    def _init_rate_limiting(self, enabled: bool) -> None:
        """Initialize rate limiting."""
        self.rate_limiting_enabled = enabled
        if enabled:
            self.rate_limiter = RateLimiter(
                persistence_path=str(self.security_dir / "rate_limits.json")
            )
            # Add default limits
            for config in create_default_rate_limits():
                self.rate_limiter.add_config(config)
        else:
            self.rate_limiter = None

    def _init_mtls(self) -> None:
        """Initialize mTLS."""
        self.mtls_manager = MTLSManager(
            cert_dir=str(self.security_dir / "certs"),
            ca_common_name="Claude Coordination CA",
        )
        # Generate CA if it doesn't exist
        ca_path = self.security_dir / "certs" / "ca.crt"
        if not ca_path.exists():
            self.mtls_manager.generate_ca()

    def _init_secret_rotation(self) -> None:
        """Initialize secret rotation."""
        self.rotation_manager = SecretRotationManager(
            storage_path=str(self.security_dir / "rotation.json"),
            on_rotation=self._handle_rotation,
        )
        # Register JWT secret for rotation
        if 'jwt_secret' not in [s['name'] for s in self.rotation_manager.list_secrets()]:
            self.rotation_manager.register_secret(
                name='jwt_secret',
                secret_type=SecretType.JWT_SECRET,
                initial_value=self.jwt_manager.secret_key,
                schedule=RotationSchedule.WEEKLY,
            )

    def _handle_rotation(self, name: str, old_value: str, new_value: str) -> None:
        """Handle secret rotation events."""
        if name == 'jwt_secret':
            # Update JWT manager with new secret
            self.jwt_manager = JWTManager(secret_key=new_value)

        if self.audit_logger:
            self.audit_logger.log(
                event_type=AuditEventType.SECURITY_KEY_ROTATED,
                actor="system",
                resource_type="secret",
                resource_id=name,
                action="rotate",
            )

    # =========================================================================
    # Authentication Methods
    # =========================================================================

    def generate_api_key(
        self,
        admin_token: str,
        name: str,
        role: Role,
        agent_id: Optional[str] = None,
        expires_in_days: Optional[int] = None,
    ) -> Tuple[str, str]:
        """
        Generate a new API key (admin only).

        Returns: (key_id, plaintext_key)
        """
        # Verify admin access
        auth_info = self._verify_token(admin_token)
        if not self.rbac_manager.check_permission(auth_info['role'], Permission.ADMIN_USERS):
            raise AuthorizationError("Admin permission required to generate API keys")

        key_id, key = self.api_key_manager.generate_key(
            name=name,
            role=role,
            agent_id=agent_id,
            expires_in_days=expires_in_days,
        )

        if self.audit_logger:
            self.audit_logger.log(
                event_type=AuditEventType.AUTH_KEY_CREATED,
                actor=auth_info['sub'],
                actor_role=auth_info['role'].value,
                resource_type="api_key",
                resource_id=key_id,
                details={'name': name, 'role': role.value},
            )

        return key_id, key

    def authenticate(self, api_key: str) -> str:
        """
        Authenticate with an API key and get a JWT token.

        Returns: JWT access token
        """
        # Rate limit authentication attempts
        if self.rate_limiter:
            try:
                self.rate_limiter.consume("auth_attempt", limit_name="auth_attempts")
            except RateLimitExceeded as e:
                if self.audit_logger:
                    self.audit_logger.log_auth_failure(
                        reason="rate_limited",
                    )
                raise AuthenticationError(f"Too many authentication attempts. Retry after {e.retry_after_seconds:.0f}s")

        # Validate API key
        key_info = self.api_key_manager.validate_key(api_key)
        if not key_info:
            if self.audit_logger:
                self.audit_logger.log_auth_failure(reason="invalid_key")
            raise AuthenticationError("Invalid API key")

        # Generate JWT token
        token = self.jwt_manager.create_access_token(
            subject=key_info.agent_id or key_info.key_id,
            role=key_info.role,
            additional_claims={'key_id': key_info.key_id},
        )

        if self.audit_logger:
            self.audit_logger.log_auth_success(
                actor=key_info.agent_id or key_info.key_id,
                actor_role=key_info.role.value,
                method="api_key",
            )

        return token

    def refresh_token(self, refresh_token: str) -> str:
        """Refresh an access token."""
        new_token = self.jwt_manager.refresh_access_token(refresh_token)
        if not new_token:
            raise AuthenticationError("Invalid or expired refresh token")
        return new_token

    def _verify_token(self, token: str) -> Dict[str, Any]:
        """Verify a JWT token and return its payload."""
        payload = self.jwt_manager.validate_token(token)
        if not payload:
            raise AuthenticationError("Invalid or expired token")

        payload['role'] = Role(payload['role'])
        return payload

    def _require_permission(self, token: str, permission: Permission) -> Dict[str, Any]:
        """Verify token and check permission."""
        auth_info = self._verify_token(token)

        if not self.rbac_manager.check_permission(
            auth_info['role'],
            permission,
            agent_id=auth_info.get('sub'),
        ):
            if self.audit_logger:
                self.audit_logger.log_access_denied(
                    actor=auth_info['sub'],
                    actor_role=auth_info['role'].value,
                    permission=permission.value,
                    resource_type="coordination",
                )
            raise AuthorizationError(f"Permission denied: {permission.value}")

        return auth_info

    def _check_rate_limit(self, identity: str, limit_name: str, role: str) -> None:
        """Check rate limit for an operation."""
        if not self.rate_limiter:
            return

        try:
            self.rate_limiter.consume(identity, limit_name=limit_name, role=role)
        except RateLimitExceeded as e:
            if self.audit_logger:
                self.audit_logger.log_rate_limited(
                    actor=identity,
                    limit_name=limit_name,
                    requests_made=0,  # Would need tracking
                    limit=0,
                )
            raise

    # =========================================================================
    # Leader Operations
    # =========================================================================

    def leader_init(self, token: str, goal: str, approach: str = "") -> None:
        """Initialize a new coordination session (leader only)."""
        auth_info = self._require_permission(token, Permission.PLAN_CREATE)

        # Sanitize inputs
        goal_result = self.sanitizer.sanitize_task_description(goal)
        approach_result = self.sanitizer.sanitize_task_description(approach)

        # Import and call original
        from coordination import leader_init
        leader_init(goal_result.sanitized_value, approach_result.sanitized_value)

        if self.audit_logger:
            self.audit_logger.log(
                event_type=AuditEventType.COORDINATION_INIT,
                actor=auth_info['sub'],
                actor_role=auth_info['role'].value,
                details={'goal': goal_result.sanitized_value[:100]},
            )

    def leader_add_task(
        self,
        token: str,
        description: str,
        priority: int = 5,
        dependencies: Optional[List[str]] = None,
        context_files: Optional[List[str]] = None,
        hints: str = "",
    ) -> str:
        """Add a new task (leader only)."""
        auth_info = self._require_permission(token, Permission.TASK_CREATE)
        self._check_rate_limit(auth_info['sub'], "task_create", auth_info['role'].value)

        # Sanitize all inputs
        sanitized = sanitize_task_input(
            description=description,
            priority=priority,
            dependencies=dependencies,
            context_files=context_files,
            hints=hints,
        )

        # Create task
        task = Task(
            id=generate_task_id(),
            description=sanitized['description'],
            status="available",
            priority=sanitized['priority'],
            dependencies=sanitized['dependencies'],
            context={
                'files': sanitized['context_files'],
                'hints': sanitized['hints'],
            } if sanitized['context_files'] or sanitized['hints'] else None,
            created_at=now_iso(),
        )

        # Encrypt if enabled
        task_dict = task.to_dict()
        if self.encryption_enabled:
            task_dict = self.encryption_manager.encrypt_task_data(task_dict)

        # Save
        data = load_tasks()
        data['tasks'].append(task_dict)
        save_tasks(data)

        log_action("leader", "ADD_TASK", f"{task.id}: {description[:50]}")

        if self.audit_logger:
            self.audit_logger.log_task_event(
                event_type=AuditEventType.TASK_CREATED,
                task_id=task.id,
                actor=auth_info['sub'],
                actor_role=auth_info['role'].value,
                details={'priority': priority},
            )

        return task.id

    def leader_status(self, token: str) -> Dict[str, Any]:
        """Get current coordination status (leader/admin only)."""
        auth_info = self._require_permission(token, Permission.TASK_READ)

        data = load_tasks()
        tasks = []
        for t in data['tasks']:
            # Decrypt if needed
            if self.encryption_enabled and t.get('_description_encrypted'):
                t = self.encryption_manager.decrypt_task_data(t)
            tasks.append(Task.from_dict(t))

        by_status = {}
        for task in tasks:
            by_status.setdefault(task.status, []).append(task)

        return {
            'total': len(tasks),
            'by_status': {k: len(v) for k, v in by_status.items()},
            'progress_percent': (
                len(by_status.get('done', [])) / len(tasks) * 100
                if tasks else 0
            ),
        }

    # =========================================================================
    # Worker Operations
    # =========================================================================

    def worker_claim(self, token: str) -> Optional[Dict[str, Any]]:
        """Claim an available task (worker only)."""
        auth_info = self._require_permission(token, Permission.TASK_CLAIM)
        self._check_rate_limit(auth_info['sub'], "task_claim", auth_info['role'].value)

        terminal_id = auth_info['sub']
        data = load_tasks()
        tasks = []
        for t in data['tasks']:
            if self.encryption_enabled and t.get('_description_encrypted'):
                t = self.encryption_manager.decrypt_task_data(t)
            tasks.append(Task.from_dict(t))

        done_ids = {t.id for t in tasks if t.status == 'done'}

        # Find available task
        available = [
            t for t in tasks
            if t.status == 'available'
            and all(dep in done_ids for dep in (t.dependencies or []))
        ]

        if not available:
            return None

        # Claim highest priority task
        task = min(available, key=lambda t: t.priority)

        # Update in data
        for i, t in enumerate(data['tasks']):
            task_id = t.get('id')
            if task_id == task.id:
                data['tasks'][i]['status'] = 'claimed'
                data['tasks'][i]['claimed_by'] = terminal_id
                data['tasks'][i]['claimed_at'] = now_iso()
                break

        save_tasks(data)

        log_action(terminal_id, "CLAIMED", task.id)

        if self.audit_logger:
            self.audit_logger.log_task_event(
                event_type=AuditEventType.TASK_CLAIMED,
                task_id=task.id,
                actor=terminal_id,
                actor_role=auth_info['role'].value,
            )

        return task.to_dict()

    def worker_complete(
        self,
        token: str,
        task_id: str,
        output: str,
        files_modified: Optional[List[str]] = None,
        files_created: Optional[List[str]] = None,
    ) -> bool:
        """Mark a task as complete (worker only)."""
        auth_info = self._require_permission(token, Permission.TASK_COMPLETE)
        terminal_id = auth_info['sub']

        # Sanitize output
        output_result = self.sanitizer.sanitize_task_description(output)

        data = load_tasks()
        task = None
        for i, t in enumerate(data['tasks']):
            if t.get('id') == task_id:
                if t.get('claimed_by') != terminal_id:
                    raise AuthorizationError(f"Task {task_id} not claimed by you")

                data['tasks'][i]['status'] = 'done'
                data['tasks'][i]['completed_at'] = now_iso()
                data['tasks'][i]['result'] = {
                    'output': output_result.sanitized_value,
                    'files_modified': files_modified or [],
                    'files_created': files_created or [],
                }

                # Encrypt result if enabled
                if self.encryption_enabled:
                    result = data['tasks'][i]['result']
                    result['output'] = self.encryption_manager.encrypt(result['output'])
                    result['_output_encrypted'] = True

                task = t
                break

        if not task:
            return False

        save_tasks(data)
        log_action(terminal_id, "COMPLETED", task_id)

        if self.audit_logger:
            self.audit_logger.log_task_event(
                event_type=AuditEventType.TASK_COMPLETED,
                task_id=task_id,
                actor=terminal_id,
                actor_role=auth_info['role'].value,
            )

        return True

    def worker_fail(self, token: str, task_id: str, reason: str) -> bool:
        """Mark a task as failed (worker only)."""
        auth_info = self._require_permission(token, Permission.TASK_COMPLETE)
        terminal_id = auth_info['sub']

        # Sanitize reason
        reason_result = self.sanitizer.sanitize_task_description(reason)

        data = load_tasks()
        task = None
        for i, t in enumerate(data['tasks']):
            if t.get('id') == task_id:
                data['tasks'][i]['status'] = 'failed'
                data['tasks'][i]['completed_at'] = now_iso()
                data['tasks'][i]['result'] = {'error': reason_result.sanitized_value}
                task = t
                break

        if not task:
            return False

        save_tasks(data)
        log_action(terminal_id, "FAILED", f"{task_id}: {reason}")

        if self.audit_logger:
            self.audit_logger.log_task_event(
                event_type=AuditEventType.TASK_FAILED,
                task_id=task_id,
                actor=terminal_id,
                actor_role=auth_info['role'].value,
                details={'reason': reason_result.sanitized_value[:100]},
            )

        return True

    # =========================================================================
    # Admin Operations
    # =========================================================================

    def rotate_secrets(self, token: str) -> Dict[str, bool]:
        """Rotate all managed secrets (admin only)."""
        auth_info = self._require_permission(token, Permission.ADMIN_ROTATE_SECRETS)
        return self.rotation_manager.rotate_all()

    def get_audit_log(
        self,
        token: str,
        limit: int = 100,
        event_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get audit log entries (admin only)."""
        auth_info = self._require_permission(token, Permission.ADMIN_AUDIT)

        if not self.audit_logger:
            return []

        event_type_enum = AuditEventType(event_type) if event_type else None
        events = self.audit_logger.query_events(
            event_type=event_type_enum,
            limit=limit,
        )

        return [e.to_dict() for e in events]

    def revoke_api_key(self, token: str, key_id: str) -> bool:
        """Revoke an API key (admin only)."""
        auth_info = self._require_permission(token, Permission.ADMIN_USERS)

        success = self.api_key_manager.revoke_key(key_id)

        if success and self.audit_logger:
            self.audit_logger.log(
                event_type=AuditEventType.AUTH_KEY_REVOKED,
                actor=auth_info['sub'],
                actor_role=auth_info['role'].value,
                resource_type="api_key",
                resource_id=key_id,
            )

        return success

    # =========================================================================
    # mTLS Operations
    # =========================================================================

    def generate_agent_certificate(
        self,
        token: str,
        agent_id: str,
        role: str = "worker",
    ) -> Tuple[str, str]:
        """Generate a certificate for an agent (admin only)."""
        if not self.mtls_manager:
            raise ValueError("mTLS is not enabled")

        auth_info = self._require_permission(token, Permission.ADMIN_CONFIG)

        cert_path, key_path = self.mtls_manager.generate_agent_cert(
            agent_id=agent_id,
            role=role,
        )

        if self.audit_logger:
            self.audit_logger.log(
                event_type=AuditEventType.SECURITY_KEY_ROTATED,
                actor=auth_info['sub'],
                actor_role=auth_info['role'].value,
                resource_type="certificate",
                resource_id=agent_id,
                action="generate",
            )

        return cert_path, key_path


# Convenience function for CLI usage
def create_secure_coordinator(**kwargs) -> SecureCoordinator:
    """Create a secure coordinator with default settings."""
    return SecureCoordinator(**kwargs)


if __name__ == "__main__":
    # Demo usage
    coordinator = SecureCoordinator()
    print("Secure coordinator initialized.")
    print("Use the generated admin key to authenticate and perform operations.")
