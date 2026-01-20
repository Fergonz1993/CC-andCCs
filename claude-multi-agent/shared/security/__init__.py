"""
Shared Security Module for Claude Multi-Agent Coordination System.

This module provides comprehensive security features that can be used
across all three options (A, B, C) of the coordination system.

Features:
- API key authentication
- JWT token support with expiration
- Role-based access control (RBAC)
- Task data encryption at rest
- Secure credential storage
- Audit logging
- Input sanitization
- Rate limiting
- mTLS support
- Secret rotation
"""

from .auth import (
    APIKeyManager,
    JWTManager,
    RBACManager,
    Role,
    Permission,
)
from .encryption import (
    EncryptionManager,
    SecureCredentialStore,
)
from .audit import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
)
from .sanitization import (
    InputSanitizer,
    SanitizationError,
)
from .rate_limiting import (
    RateLimiter,
    RateLimitExceeded,
)
from .mtls import (
    MTLSManager,
    CertificateInfo,
)
from .rotation import (
    SecretRotationManager,
    RotationSchedule,
)

__all__ = [
    # Authentication
    "APIKeyManager",
    "JWTManager",
    "RBACManager",
    "Role",
    "Permission",
    # Encryption
    "EncryptionManager",
    "SecureCredentialStore",
    # Audit
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    # Sanitization
    "InputSanitizer",
    "SanitizationError",
    # Rate Limiting
    "RateLimiter",
    "RateLimitExceeded",
    # mTLS
    "MTLSManager",
    "CertificateInfo",
    # Rotation
    "SecretRotationManager",
    "RotationSchedule",
]

__version__ = "1.0.0"
