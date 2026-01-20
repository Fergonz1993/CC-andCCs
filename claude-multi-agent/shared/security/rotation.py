"""
Secret Rotation Module for Claude Multi-Agent Coordination System.

Provides automated secret rotation mechanisms.
"""

import json
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
import time


class RotationSchedule(str, Enum):
    """Predefined rotation schedules."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"

    def to_timedelta(self) -> timedelta:
        """Convert schedule to timedelta."""
        mapping = {
            RotationSchedule.HOURLY: timedelta(hours=1),
            RotationSchedule.DAILY: timedelta(days=1),
            RotationSchedule.WEEKLY: timedelta(weeks=1),
            RotationSchedule.MONTHLY: timedelta(days=30),
        }
        return mapping.get(self, timedelta(days=1))


class SecretType(str, Enum):
    """Types of secrets that can be rotated."""
    API_KEY = "api_key"
    JWT_SECRET = "jwt_secret"
    ENCRYPTION_KEY = "encryption_key"
    CERTIFICATE = "certificate"
    PASSWORD = "password"
    TOKEN = "token"
    CUSTOM = "custom"


@dataclass
class SecretMetadata:
    """Metadata about a managed secret."""
    name: str
    secret_type: SecretType
    created_at: str
    last_rotated: Optional[str] = None
    next_rotation: Optional[str] = None
    rotation_schedule: RotationSchedule = RotationSchedule.DAILY
    rotation_count: int = 0
    is_active: bool = True
    custom_interval_seconds: Optional[int] = None
    version: int = 1
    previous_versions: List[str] = field(default_factory=list)  # Keep track for grace period
    grace_period_seconds: int = 300  # 5 minutes to transition to new secret
    auto_rotate: bool = True


@dataclass
class RotationEvent:
    """Record of a rotation event."""
    secret_name: str
    old_version: int
    new_version: int
    timestamp: str
    success: bool
    error: Optional[str] = None
    triggered_by: str = "scheduled"  # scheduled, manual, emergency


class SecretRotationManager:
    """
    Manages automatic secret rotation.

    Features:
    - Scheduled rotation of secrets
    - Multiple secret types support
    - Grace period for old secrets
    - Rotation callbacks
    - Rotation history
    - Manual rotation trigger
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        on_rotation: Optional[Callable[[str, str, str], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
    ):
        """
        Initialize the secret rotation manager.

        Args:
            storage_path: Path to store rotation metadata
            on_rotation: Callback(secret_name, old_value, new_value) on successful rotation
            on_error: Callback(secret_name, exception) on rotation error
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.on_rotation = on_rotation
        self.on_error = on_error

        self._secrets: Dict[str, SecretMetadata] = {}
        self._secret_values: Dict[str, str] = {}  # Current values
        self._previous_values: Dict[str, List[tuple]] = {}  # (value, expires_at)
        self._rotation_history: List[RotationEvent] = []
        self._generators: Dict[SecretType, Callable[[], str]] = {}
        self._lock = threading.Lock()
        self._rotation_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Register default generators
        self._register_default_generators()

        # Load existing state
        if self.storage_path:
            self._load_state()

    def _register_default_generators(self) -> None:
        """Register default secret generators."""
        self._generators = {
            SecretType.API_KEY: lambda: f"mca_{secrets.token_hex(32)}",
            SecretType.JWT_SECRET: lambda: secrets.token_hex(64),
            SecretType.ENCRYPTION_KEY: lambda: secrets.token_hex(32),
            SecretType.PASSWORD: lambda: secrets.token_urlsafe(24),
            SecretType.TOKEN: lambda: secrets.token_urlsafe(32),
        }

    def register_generator(
        self,
        secret_type: SecretType,
        generator: Callable[[], str],
    ) -> None:
        """Register a custom secret generator."""
        self._generators[secret_type] = generator

    def register_secret(
        self,
        name: str,
        secret_type: SecretType,
        initial_value: Optional[str] = None,
        schedule: RotationSchedule = RotationSchedule.DAILY,
        custom_interval_seconds: Optional[int] = None,
        grace_period_seconds: int = 300,
        auto_rotate: bool = True,
    ) -> str:
        """
        Register a new secret for management.

        Returns the initial secret value.
        """
        with self._lock:
            if name in self._secrets:
                raise ValueError(f"Secret '{name}' already registered")

            # Generate initial value if not provided
            if initial_value is None:
                generator = self._generators.get(secret_type)
                if generator:
                    initial_value = generator()
                else:
                    initial_value = secrets.token_hex(32)

            now = datetime.now()
            if schedule == RotationSchedule.CUSTOM and custom_interval_seconds:
                next_rotation = now + timedelta(seconds=custom_interval_seconds)
            else:
                next_rotation = now + schedule.to_timedelta()

            metadata = SecretMetadata(
                name=name,
                secret_type=secret_type,
                created_at=now.isoformat(),
                next_rotation=next_rotation.isoformat(),
                rotation_schedule=schedule,
                custom_interval_seconds=custom_interval_seconds,
                grace_period_seconds=grace_period_seconds,
                auto_rotate=auto_rotate,
            )

            self._secrets[name] = metadata
            self._secret_values[name] = initial_value
            self._previous_values[name] = []

            self._save_state()
            return initial_value

    def get_secret(
        self,
        name: str,
        include_previous: bool = False,
    ) -> Optional[str]:
        """
        Get a secret value.

        Args:
            name: Secret name
            include_previous: If True, returns (current, [previous_values])
        """
        with self._lock:
            if name not in self._secrets:
                return None

            current = self._secret_values.get(name)

            if include_previous:
                # Filter out expired previous values
                now = time.time()
                valid_previous = [
                    val for val, expires_at in self._previous_values.get(name, [])
                    if expires_at > now
                ]
                return current, valid_previous

            return current

    def validate_secret(self, name: str, value: str) -> bool:
        """
        Validate if a secret value is valid (current or in grace period).
        """
        with self._lock:
            if name not in self._secrets:
                return False

            # Check current value
            if self._secret_values.get(name) == value:
                return True

            # Check previous values in grace period
            now = time.time()
            for prev_value, expires_at in self._previous_values.get(name, []):
                if expires_at > now and prev_value == value:
                    return True

            return False

    def rotate_secret(
        self,
        name: str,
        new_value: Optional[str] = None,
        triggered_by: str = "manual",
    ) -> Optional[str]:
        """
        Rotate a secret to a new value.

        Args:
            name: Secret name
            new_value: New value (generated if not provided)
            triggered_by: What triggered this rotation

        Returns:
            The new secret value, or None if rotation failed
        """
        with self._lock:
            metadata = self._secrets.get(name)
            if not metadata:
                return None

            old_value = self._secret_values.get(name)
            old_version = metadata.version

            # Generate new value if not provided
            if new_value is None:
                generator = self._generators.get(metadata.secret_type)
                if generator:
                    new_value = generator()
                else:
                    new_value = secrets.token_hex(32)

            try:
                # Move current to previous with grace period
                if old_value:
                    grace_expires = time.time() + metadata.grace_period_seconds
                    self._previous_values.setdefault(name, []).append(
                        (old_value, grace_expires)
                    )
                    # Keep only last 5 versions in grace period
                    self._previous_values[name] = self._previous_values[name][-5:]

                # Update secret
                self._secret_values[name] = new_value
                metadata.version += 1
                metadata.last_rotated = datetime.now().isoformat()
                metadata.rotation_count += 1

                # Schedule next rotation
                if metadata.rotation_schedule == RotationSchedule.CUSTOM:
                    interval = metadata.custom_interval_seconds or 86400
                    metadata.next_rotation = (
                        datetime.now() + timedelta(seconds=interval)
                    ).isoformat()
                else:
                    metadata.next_rotation = (
                        datetime.now() + metadata.rotation_schedule.to_timedelta()
                    ).isoformat()

                # Record event
                event = RotationEvent(
                    secret_name=name,
                    old_version=old_version,
                    new_version=metadata.version,
                    timestamp=datetime.now().isoformat(),
                    success=True,
                    triggered_by=triggered_by,
                )
                self._rotation_history.append(event)

                # Keep only last 100 events
                if len(self._rotation_history) > 100:
                    self._rotation_history = self._rotation_history[-100:]

                self._save_state()

                # Callback
                if self.on_rotation:
                    try:
                        self.on_rotation(name, old_value, new_value)
                    except Exception:
                        pass

                return new_value

            except Exception as e:
                # Record failure
                event = RotationEvent(
                    secret_name=name,
                    old_version=old_version,
                    new_version=old_version,
                    timestamp=datetime.now().isoformat(),
                    success=False,
                    error=str(e),
                    triggered_by=triggered_by,
                )
                self._rotation_history.append(event)

                if self.on_error:
                    try:
                        self.on_error(name, e)
                    except Exception:
                        pass

                return None

    def force_rotation(self, name: str) -> Optional[str]:
        """Force immediate rotation of a secret."""
        return self.rotate_secret(name, triggered_by="emergency")

    def rotate_all(self) -> Dict[str, bool]:
        """Rotate all registered secrets immediately."""
        results = {}
        for name in list(self._secrets.keys()):
            new_value = self.rotate_secret(name, triggered_by="manual")
            results[name] = new_value is not None
        return results

    def get_metadata(self, name: str) -> Optional[SecretMetadata]:
        """Get metadata for a secret."""
        return self._secrets.get(name)

    def list_secrets(self) -> List[Dict[str, Any]]:
        """List all managed secrets (without values)."""
        return [
            {
                'name': m.name,
                'secret_type': m.secret_type.value,
                'created_at': m.created_at,
                'last_rotated': m.last_rotated,
                'next_rotation': m.next_rotation,
                'rotation_schedule': m.rotation_schedule.value,
                'rotation_count': m.rotation_count,
                'version': m.version,
                'auto_rotate': m.auto_rotate,
            }
            for m in self._secrets.values()
        ]

    def get_rotation_history(
        self,
        secret_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get rotation history."""
        events = self._rotation_history
        if secret_name:
            events = [e for e in events if e.secret_name == secret_name]

        return [
            {
                'secret_name': e.secret_name,
                'old_version': e.old_version,
                'new_version': e.new_version,
                'timestamp': e.timestamp,
                'success': e.success,
                'error': e.error,
                'triggered_by': e.triggered_by,
            }
            for e in events[-limit:]
        ]

    def unregister_secret(self, name: str) -> bool:
        """Unregister a secret from management."""
        with self._lock:
            if name in self._secrets:
                del self._secrets[name]
                if name in self._secret_values:
                    del self._secret_values[name]
                if name in self._previous_values:
                    del self._previous_values[name]
                self._save_state()
                return True
            return False

    def start_auto_rotation(self, check_interval_seconds: int = 60) -> None:
        """
        Start the automatic rotation background thread.

        Args:
            check_interval_seconds: How often to check for needed rotations
        """
        if self._rotation_thread and self._rotation_thread.is_alive():
            return

        self._stop_event.clear()
        self._rotation_thread = threading.Thread(
            target=self._rotation_loop,
            args=(check_interval_seconds,),
            daemon=True,
        )
        self._rotation_thread.start()

    def stop_auto_rotation(self) -> None:
        """Stop the automatic rotation background thread."""
        self._stop_event.set()
        if self._rotation_thread:
            self._rotation_thread.join(timeout=5)

    def _rotation_loop(self, check_interval: int) -> None:
        """Background loop that checks for and performs rotations."""
        while not self._stop_event.is_set():
            try:
                self._check_and_rotate()
            except Exception:
                pass  # Don't let errors stop the loop

            # Wait for next check
            self._stop_event.wait(check_interval)

    def _check_and_rotate(self) -> None:
        """Check for secrets that need rotation and rotate them."""
        now = datetime.now()

        with self._lock:
            for name, metadata in list(self._secrets.items()):
                if not metadata.auto_rotate:
                    continue
                if not metadata.next_rotation:
                    continue

                next_rotation = datetime.fromisoformat(metadata.next_rotation)
                if now >= next_rotation:
                    # Release lock during rotation
                    pass

        # Rotate outside the lock
        for name, metadata in list(self._secrets.items()):
            if not metadata.auto_rotate:
                continue
            if not metadata.next_rotation:
                continue

            next_rotation = datetime.fromisoformat(metadata.next_rotation)
            if now >= next_rotation:
                self.rotate_secret(name, triggered_by="scheduled")

    def _load_state(self) -> None:
        """Load state from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            for name, meta_data in data.get('secrets', {}).items():
                self._secrets[name] = SecretMetadata(
                    name=meta_data['name'],
                    secret_type=SecretType(meta_data['secret_type']),
                    created_at=meta_data['created_at'],
                    last_rotated=meta_data.get('last_rotated'),
                    next_rotation=meta_data.get('next_rotation'),
                    rotation_schedule=RotationSchedule(meta_data.get('rotation_schedule', 'daily')),
                    rotation_count=meta_data.get('rotation_count', 0),
                    is_active=meta_data.get('is_active', True),
                    custom_interval_seconds=meta_data.get('custom_interval_seconds'),
                    version=meta_data.get('version', 1),
                    previous_versions=meta_data.get('previous_versions', []),
                    grace_period_seconds=meta_data.get('grace_period_seconds', 300),
                    auto_rotate=meta_data.get('auto_rotate', True),
                )

            self._secret_values = data.get('values', {})

            for event_data in data.get('history', []):
                self._rotation_history.append(RotationEvent(
                    secret_name=event_data['secret_name'],
                    old_version=event_data['old_version'],
                    new_version=event_data['new_version'],
                    timestamp=event_data['timestamp'],
                    success=event_data['success'],
                    error=event_data.get('error'),
                    triggered_by=event_data.get('triggered_by', 'unknown'),
                ))

        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            pass

    def _save_state(self) -> None:
        """Save state to storage."""
        if not self.storage_path:
            return

        data = {
            'secrets': {
                name: {
                    'name': m.name,
                    'secret_type': m.secret_type.value,
                    'created_at': m.created_at,
                    'last_rotated': m.last_rotated,
                    'next_rotation': m.next_rotation,
                    'rotation_schedule': m.rotation_schedule.value,
                    'rotation_count': m.rotation_count,
                    'is_active': m.is_active,
                    'custom_interval_seconds': m.custom_interval_seconds,
                    'version': m.version,
                    'previous_versions': m.previous_versions,
                    'grace_period_seconds': m.grace_period_seconds,
                    'auto_rotate': m.auto_rotate,
                }
                for name, m in self._secrets.items()
            },
            'values': self._secret_values,
            'history': [
                {
                    'secret_name': e.secret_name,
                    'old_version': e.old_version,
                    'new_version': e.new_version,
                    'timestamp': e.timestamp,
                    'success': e.success,
                    'error': e.error,
                    'triggered_by': e.triggered_by,
                }
                for e in self._rotation_history[-100:]
            ],
        }

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def update_schedule(
        self,
        name: str,
        schedule: RotationSchedule,
        custom_interval_seconds: Optional[int] = None,
    ) -> bool:
        """Update the rotation schedule for a secret."""
        with self._lock:
            metadata = self._secrets.get(name)
            if not metadata:
                return False

            metadata.rotation_schedule = schedule
            metadata.custom_interval_seconds = custom_interval_seconds

            # Update next rotation time
            if schedule == RotationSchedule.CUSTOM and custom_interval_seconds:
                metadata.next_rotation = (
                    datetime.now() + timedelta(seconds=custom_interval_seconds)
                ).isoformat()
            else:
                metadata.next_rotation = (
                    datetime.now() + schedule.to_timedelta()
                ).isoformat()

            self._save_state()
            return True

    def set_auto_rotate(self, name: str, enabled: bool) -> bool:
        """Enable or disable auto-rotation for a secret."""
        with self._lock:
            metadata = self._secrets.get(name)
            if not metadata:
                return False

            metadata.auto_rotate = enabled
            self._save_state()
            return True
