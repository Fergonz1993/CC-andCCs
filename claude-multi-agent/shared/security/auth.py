"""
Authentication Module for Claude Multi-Agent Coordination System.

Provides:
- API key generation and validation
- JWT token support with expiration
- Role-based access control (RBAC)
"""

import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Set, Any
import base64

# Try to import JWT library, fallback to simple implementation if not available
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False


class Permission(str, Enum):
    """Permissions for RBAC."""
    # Task operations
    TASK_CREATE = "task:create"
    TASK_READ = "task:read"
    TASK_CLAIM = "task:claim"
    TASK_COMPLETE = "task:complete"
    TASK_CANCEL = "task:cancel"
    TASK_DELETE = "task:delete"

    # Discovery operations
    DISCOVERY_CREATE = "discovery:create"
    DISCOVERY_READ = "discovery:read"

    # Agent operations
    AGENT_REGISTER = "agent:register"
    AGENT_HEARTBEAT = "agent:heartbeat"
    AGENT_READ = "agent:read"
    AGENT_DELETE = "agent:delete"

    # Admin operations
    ADMIN_CONFIG = "admin:config"
    ADMIN_USERS = "admin:users"
    ADMIN_AUDIT = "admin:audit"
    ADMIN_ROTATE_SECRETS = "admin:rotate_secrets"

    # Plan operations
    PLAN_CREATE = "plan:create"
    PLAN_READ = "plan:read"
    PLAN_UPDATE = "plan:update"


class Role(str, Enum):
    """Predefined roles for the system."""
    ADMIN = "admin"
    LEADER = "leader"
    WORKER = "worker"
    READONLY = "readonly"


# Default permissions per role
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: set(Permission),  # All permissions
    Role.LEADER: {
        Permission.TASK_CREATE,
        Permission.TASK_READ,
        Permission.TASK_CANCEL,
        Permission.DISCOVERY_CREATE,
        Permission.DISCOVERY_READ,
        Permission.AGENT_REGISTER,
        Permission.AGENT_HEARTBEAT,
        Permission.AGENT_READ,
        Permission.PLAN_CREATE,
        Permission.PLAN_READ,
        Permission.PLAN_UPDATE,
    },
    Role.WORKER: {
        Permission.TASK_READ,
        Permission.TASK_CLAIM,
        Permission.TASK_COMPLETE,
        Permission.DISCOVERY_CREATE,
        Permission.DISCOVERY_READ,
        Permission.AGENT_REGISTER,
        Permission.AGENT_HEARTBEAT,
        Permission.PLAN_READ,
    },
    Role.READONLY: {
        Permission.TASK_READ,
        Permission.DISCOVERY_READ,
        Permission.AGENT_READ,
        Permission.PLAN_READ,
    },
}


@dataclass
class APIKey:
    """Represents an API key with metadata."""
    key_id: str
    key_hash: str  # Store hash, not the actual key
    name: str
    role: Role
    agent_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    expires_at: Optional[str] = None
    last_used: Optional[str] = None
    is_revoked: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "APIKey":
        data['role'] = Role(data['role'])
        return cls(**data)


class APIKeyManager:
    """
    Manages API keys for authentication.

    Features:
    - Generate secure API keys
    - Validate keys without storing plaintext
    - Revoke keys
    - Track key usage
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self._keys: Dict[str, APIKey] = {}
        self._key_lookup: Dict[str, str] = {}  # prefix -> key_id for fast lookup

        if self.storage_path and self.storage_path.exists():
            self._load_keys()

    def _load_keys(self) -> None:
        """Load keys from storage."""
        if not self.storage_path:
            return
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            for key_data in data.get('keys', []):
                key = APIKey.from_dict(key_data)
                self._keys[key.key_id] = key
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    def _save_keys(self) -> None:
        """Save keys to storage."""
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump({'keys': [k.to_dict() for k in self._keys.values()]}, f, indent=2)

    def generate_key(
        self,
        name: str,
        role: Role,
        agent_id: Optional[str] = None,
        expires_in_days: Optional[int] = None,
    ) -> tuple[str, str]:
        """
        Generate a new API key.

        Returns: (key_id, plaintext_key)
        """
        # Generate secure key: prefix.random_bytes
        key_id = f"mca_{secrets.token_hex(8)}"
        secret_part = secrets.token_urlsafe(32)
        plaintext_key = f"{key_id}.{secret_part}"

        # Store hash of the key
        key_hash = hashlib.sha256(plaintext_key.encode()).hexdigest()

        expires_at = None
        if expires_in_days:
            expires_at = (datetime.now() + timedelta(days=expires_in_days)).isoformat()

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            role=role,
            agent_id=agent_id,
            expires_at=expires_at,
        )

        self._keys[key_id] = api_key
        self._key_lookup[key_id] = key_id
        self._save_keys()

        return key_id, plaintext_key

    def validate_key(self, plaintext_key: str) -> Optional[APIKey]:
        """
        Validate an API key.

        Returns the APIKey if valid, None otherwise.
        """
        if not plaintext_key or '.' not in plaintext_key:
            return None

        try:
            key_id = plaintext_key.split('.')[0]
        except (IndexError, ValueError):
            return None

        api_key = self._keys.get(key_id)
        if not api_key:
            return None

        # Check if revoked
        if api_key.is_revoked:
            return None

        # Check expiration
        if api_key.expires_at:
            if datetime.fromisoformat(api_key.expires_at) < datetime.now():
                return None

        # Validate hash
        key_hash = hashlib.sha256(plaintext_key.encode()).hexdigest()
        if not hmac.compare_digest(key_hash, api_key.key_hash):
            return None

        # Update last used
        api_key.last_used = datetime.now().isoformat()
        self._save_keys()

        return api_key

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id not in self._keys:
            return False
        self._keys[key_id].is_revoked = True
        self._save_keys()
        return True

    def list_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (without hashes)."""
        return [
            {
                'key_id': k.key_id,
                'name': k.name,
                'role': k.role.value,
                'agent_id': k.agent_id,
                'created_at': k.created_at,
                'expires_at': k.expires_at,
                'last_used': k.last_used,
                'is_revoked': k.is_revoked,
            }
            for k in self._keys.values()
        ]


@dataclass
class JWTPayload:
    """JWT token payload."""
    sub: str  # Subject (agent_id or user_id)
    role: Role
    permissions: List[str]
    exp: int  # Expiration timestamp
    iat: int  # Issued at timestamp
    jti: str  # JWT ID for revocation


class JWTManager:
    """
    Manages JWT tokens for authentication.

    Features:
    - Generate JWT tokens with expiration
    - Validate and decode tokens
    - Token refresh
    - Token revocation
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
    ):
        self.secret_key = secret_key or os.environ.get('JWT_SECRET_KEY') or secrets.token_hex(32)
        self.algorithm = algorithm
        self.access_token_expire = timedelta(minutes=access_token_expire_minutes)
        self.refresh_token_expire = timedelta(days=refresh_token_expire_days)
        self._revoked_tokens: Set[str] = set()

    def create_access_token(
        self,
        subject: str,
        role: Role,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create an access token."""
        now = datetime.utcnow()
        exp = now + self.access_token_expire
        jti = secrets.token_hex(16)

        payload = {
            'sub': subject,
            'role': role.value,
            'permissions': [p.value for p in ROLE_PERMISSIONS.get(role, set())],
            'exp': int(exp.timestamp()),
            'iat': int(now.timestamp()),
            'jti': jti,
            'type': 'access',
        }

        if additional_claims:
            payload.update(additional_claims)

        if JWT_AVAILABLE:
            return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        else:
            return self._simple_jwt_encode(payload)

    def create_refresh_token(self, subject: str, role: Role) -> str:
        """Create a refresh token."""
        now = datetime.utcnow()
        exp = now + self.refresh_token_expire
        jti = secrets.token_hex(16)

        payload = {
            'sub': subject,
            'role': role.value,
            'exp': int(exp.timestamp()),
            'iat': int(now.timestamp()),
            'jti': jti,
            'type': 'refresh',
        }

        if JWT_AVAILABLE:
            return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        else:
            return self._simple_jwt_encode(payload)

    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate a JWT token.

        Returns the decoded payload if valid, None otherwise.
        """
        try:
            if JWT_AVAILABLE:
                payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            else:
                payload = self._simple_jwt_decode(token)

            # Check if revoked
            if payload.get('jti') in self._revoked_tokens:
                return None

            # Check expiration
            if payload.get('exp', 0) < time.time():
                return None

            return payload

        except Exception:
            return None

    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """
        Generate a new access token using a refresh token.

        Returns new access token or None if refresh token is invalid.
        """
        payload = self.validate_token(refresh_token)
        if not payload or payload.get('type') != 'refresh':
            return None

        return self.create_access_token(
            subject=payload['sub'],
            role=Role(payload['role']),
        )

    def revoke_token(self, token: str) -> bool:
        """Revoke a token by its JTI."""
        payload = self.validate_token(token)
        if payload and 'jti' in payload:
            self._revoked_tokens.add(payload['jti'])
            return True
        return False

    def _simple_jwt_encode(self, payload: Dict[str, Any]) -> str:
        """Simple JWT encoding without external library."""
        header = {'alg': 'HS256', 'typ': 'JWT'}
        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip('=')
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip('=')

        message = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip('=')

        return f"{message}.{signature_b64}"

    def _simple_jwt_decode(self, token: str) -> Dict[str, Any]:
        """Simple JWT decoding without external library."""
        parts = token.split('.')
        if len(parts) != 3:
            raise ValueError("Invalid token format")

        header_b64, payload_b64, signature_b64 = parts

        # Verify signature
        message = f"{header_b64}.{payload_b64}"
        expected_signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        expected_sig_b64 = base64.urlsafe_b64encode(expected_signature).decode().rstrip('=')

        if not hmac.compare_digest(signature_b64, expected_sig_b64):
            raise ValueError("Invalid signature")

        # Decode payload
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += '=' * padding

        payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode())
        return payload


class RBACManager:
    """
    Role-Based Access Control Manager.

    Features:
    - Role and permission management
    - Permission checks
    - Custom role definitions
    """

    def __init__(self):
        self._role_permissions: Dict[str, Set[Permission]] = {
            role.value: perms for role, perms in ROLE_PERMISSIONS.items()
        }
        self._custom_permissions: Dict[str, Set[Permission]] = {}  # agent_id -> extra permissions

    def has_permission(self, role: Role, permission: Permission) -> bool:
        """Check if a role has a specific permission."""
        role_perms = self._role_permissions.get(role.value, set())
        return permission in role_perms

    def check_permission(
        self,
        role: Role,
        permission: Permission,
        agent_id: Optional[str] = None,
    ) -> bool:
        """
        Check if the role/agent has the required permission.

        Also checks custom permissions assigned to the agent.
        """
        if self.has_permission(role, permission):
            return True

        if agent_id:
            custom_perms = self._custom_permissions.get(agent_id, set())
            if permission in custom_perms:
                return True

        return False

    def grant_permission(self, agent_id: str, permission: Permission) -> None:
        """Grant an additional permission to a specific agent."""
        if agent_id not in self._custom_permissions:
            self._custom_permissions[agent_id] = set()
        self._custom_permissions[agent_id].add(permission)

    def revoke_permission(self, agent_id: str, permission: Permission) -> bool:
        """Revoke a custom permission from an agent."""
        if agent_id in self._custom_permissions:
            self._custom_permissions[agent_id].discard(permission)
            return True
        return False

    def get_permissions(self, role: Role, agent_id: Optional[str] = None) -> Set[Permission]:
        """Get all permissions for a role, including custom ones for the agent."""
        perms = self._role_permissions.get(role.value, set()).copy()
        if agent_id:
            perms.update(self._custom_permissions.get(agent_id, set()))
        return perms

    def add_custom_role(self, role_name: str, permissions: Set[Permission]) -> None:
        """Add a custom role with specific permissions."""
        self._role_permissions[role_name] = permissions

    def get_role_permissions(self, role: Role) -> List[str]:
        """Get list of permission names for a role."""
        return [p.value for p in self._role_permissions.get(role.value, set())]
