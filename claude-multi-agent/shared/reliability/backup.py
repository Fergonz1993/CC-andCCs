"""
Backup and Restore Functionality (adv-rel-006)

Implements backup and restore mechanisms for coordination state.
Supports automatic backups, rolling backups, and restore points.
"""

import json
import shutil
import gzip
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class BackupType(str, Enum):
    """Types of backups."""
    FULL = "full"             # Complete state backup
    INCREMENTAL = "incremental"  # Only changes since last backup
    SNAPSHOT = "snapshot"     # Point-in-time snapshot


@dataclass
class BackupConfig:
    """Configuration for backup behavior."""
    backup_dir: str = ".coordination/backups"
    max_backups: int = 10              # Rolling backup window
    auto_backup_interval: float = 300.0  # 5 minutes
    compress: bool = True              # Compress backups
    include_logs: bool = False         # Include log files
    encryption_key: Optional[str] = None  # Encryption (not implemented)


@dataclass
class BackupInfo:
    """Information about a backup."""
    id: str
    type: BackupType
    filepath: str
    size_bytes: int
    created_at: datetime
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "filepath": self.filepath,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
            "checksum": self.checksum,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackupInfo":
        return cls(
            id=data["id"],
            type=BackupType(data["type"]),
            filepath=data["filepath"],
            size_bytes=data["size_bytes"],
            created_at=datetime.fromisoformat(data["created_at"]),
            checksum=data["checksum"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class RestoreResult:
    """Result of a restore operation."""
    success: bool
    backup_id: str
    restored_at: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    files_restored: List[str] = field(default_factory=list)


class BackupManager:
    """
    Manages backup and restore operations.

    Features:
    - Full and incremental backups
    - Compression support
    - Rolling backup window
    - Automatic backup scheduling
    - Restore verification
    """

    def __init__(
        self,
        config: Optional[BackupConfig] = None,
        on_backup: Optional[Callable[[BackupInfo], None]] = None,
        on_restore: Optional[Callable[[RestoreResult], None]] = None,
    ):
        """
        Initialize backup manager.

        Args:
            config: Backup configuration
            on_backup: Callback after successful backup
            on_restore: Callback after restore
        """
        self.config = config or BackupConfig()
        self.on_backup = on_backup
        self.on_restore = on_restore

        self._backup_dir = Path(self.config.backup_dir)
        self._backup_dir.mkdir(parents=True, exist_ok=True)

        self._manifest_file = self._backup_dir / "manifest.json"
        self._backups: List[BackupInfo] = self._load_manifest()

        self._auto_backup_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    def create_backup(
        self,
        state: Dict[str, Any],
        backup_type: BackupType = BackupType.FULL,
        description: str = "",
    ) -> BackupInfo:
        """
        Create a new backup.

        Args:
            state: State data to backup
            backup_type: Type of backup
            description: Optional description

        Returns:
            BackupInfo for the created backup
        """
        with self._lock:
            timestamp = datetime.now()
            backup_id = f"backup-{timestamp.strftime('%Y%m%d-%H%M%S')}"

            # Prepare backup data
            backup_data = {
                "version": "1.0",
                "created_at": timestamp.isoformat(),
                "type": backup_type.value,
                "description": description,
                "state": state,
            }

            # Serialize
            json_data = json.dumps(backup_data, indent=2, default=str)
            data_bytes = json_data.encode('utf-8')

            # Determine filename
            if self.config.compress:
                filename = f"{backup_id}.json.gz"
                filepath = self._backup_dir / filename
                with gzip.open(filepath, 'wb') as f:
                    f.write(data_bytes)
            else:
                filename = f"{backup_id}.json"
                filepath = self._backup_dir / filename
                filepath.write_bytes(data_bytes)

            # Compute checksum
            checksum = hashlib.sha256(data_bytes).hexdigest()[:16]

            # Create backup info
            backup_info = BackupInfo(
                id=backup_id,
                type=backup_type,
                filepath=str(filepath),
                size_bytes=filepath.stat().st_size,
                created_at=timestamp,
                checksum=checksum,
                metadata={
                    "description": description,
                    "compressed": self.config.compress,
                    "task_count": len(state.get("tasks", [])),
                },
            )

            self._backups.append(backup_info)
            self._save_manifest()

            # Enforce rolling window
            self._cleanup_old_backups()

            logger.info(f"Created backup: {backup_id} ({backup_info.size_bytes} bytes)")

            if self.on_backup:
                try:
                    self.on_backup(backup_info)
                except Exception as e:
                    logger.error(f"Backup callback error: {e}")

            return backup_info

    def restore_backup(
        self,
        backup_id: str,
        verify_checksum: bool = True,
    ) -> tuple[Optional[Dict[str, Any]], RestoreResult]:
        """
        Restore from a backup.

        Args:
            backup_id: ID of backup to restore
            verify_checksum: Whether to verify checksum before restoring

        Returns:
            Tuple of (restored state data, RestoreResult)
        """
        with self._lock:
            # Find backup
            backup_info = None
            for b in self._backups:
                if b.id == backup_id:
                    backup_info = b
                    break

            if not backup_info:
                result = RestoreResult(
                    success=False,
                    backup_id=backup_id,
                    error=f"Backup not found: {backup_id}",
                )
                return None, result

            try:
                filepath = Path(backup_info.filepath)

                # Read backup file
                if backup_info.metadata.get("compressed", False):
                    with gzip.open(filepath, 'rb') as f:
                        data_bytes = f.read()
                else:
                    data_bytes = filepath.read_bytes()

                # Verify checksum
                if verify_checksum:
                    actual_checksum = hashlib.sha256(data_bytes).hexdigest()[:16]
                    if actual_checksum != backup_info.checksum:
                        result = RestoreResult(
                            success=False,
                            backup_id=backup_id,
                            error=f"Checksum mismatch: expected {backup_info.checksum}, got {actual_checksum}",
                        )
                        return None, result

                # Parse backup data
                backup_data = json.loads(data_bytes.decode('utf-8'))
                state = backup_data.get("state", {})

                result = RestoreResult(
                    success=True,
                    backup_id=backup_id,
                    files_restored=[str(filepath)],
                )

                logger.info(f"Restored backup: {backup_id}")

                if self.on_restore:
                    try:
                        self.on_restore(result)
                    except Exception as e:
                        logger.error(f"Restore callback error: {e}")

                return state, result

            except Exception as e:
                result = RestoreResult(
                    success=False,
                    backup_id=backup_id,
                    error=str(e),
                )
                logger.error(f"Restore failed: {e}")
                return None, result

    def list_backups(self) -> List[BackupInfo]:
        """List all available backups."""
        return sorted(self._backups, key=lambda b: b.created_at, reverse=True)

    def get_latest_backup(self) -> Optional[BackupInfo]:
        """Get the most recent backup."""
        if not self._backups:
            return None
        return max(self._backups, key=lambda b: b.created_at)

    def delete_backup(self, backup_id: str) -> bool:
        """Delete a specific backup."""
        with self._lock:
            for i, backup in enumerate(self._backups):
                if backup.id == backup_id:
                    try:
                        Path(backup.filepath).unlink(missing_ok=True)
                        self._backups.pop(i)
                        self._save_manifest()
                        logger.info(f"Deleted backup: {backup_id}")
                        return True
                    except Exception as e:
                        logger.error(f"Failed to delete backup: {e}")
                        return False
            return False

    def start_auto_backup(
        self,
        get_state: Callable[[], Dict[str, Any]],
    ) -> None:
        """
        Start automatic backup scheduling.

        Args:
            get_state: Function that returns current state to backup
        """
        if self._running:
            return

        self._running = True
        self._auto_backup_thread = threading.Thread(
            target=self._auto_backup_loop,
            args=(get_state,),
            daemon=True,
        )
        self._auto_backup_thread.start()
        logger.info(f"Started auto-backup (interval: {self.config.auto_backup_interval}s)")

    def stop_auto_backup(self) -> None:
        """Stop automatic backup scheduling."""
        self._running = False
        if self._auto_backup_thread:
            self._auto_backup_thread.join(timeout=5.0)
            self._auto_backup_thread = None
        logger.info("Stopped auto-backup")

    def _auto_backup_loop(self, get_state: Callable[[], Dict[str, Any]]) -> None:
        """Background loop for automatic backups."""
        import time

        while self._running:
            try:
                state = get_state()
                self.create_backup(state, BackupType.FULL, "Auto backup")
            except Exception as e:
                logger.error(f"Auto backup failed: {e}")

            # Sleep in small intervals for responsiveness
            for _ in range(int(self.config.auto_backup_interval)):
                if not self._running:
                    break
                time.sleep(1)

    def _cleanup_old_backups(self) -> None:
        """Remove old backups beyond rolling window."""
        if len(self._backups) <= self.config.max_backups:
            return

        # Sort by creation time
        sorted_backups = sorted(self._backups, key=lambda b: b.created_at)

        # Remove oldest backups
        while len(sorted_backups) > self.config.max_backups:
            oldest = sorted_backups.pop(0)
            try:
                Path(oldest.filepath).unlink(missing_ok=True)
                self._backups.remove(oldest)
                logger.debug(f"Removed old backup: {oldest.id}")
            except Exception as e:
                logger.error(f"Failed to remove old backup: {e}")

        self._save_manifest()

    def _load_manifest(self) -> List[BackupInfo]:
        """Load backup manifest from disk."""
        if not self._manifest_file.exists():
            return []

        try:
            with open(self._manifest_file, 'r') as f:
                data = json.load(f)
            return [BackupInfo.from_dict(b) for b in data.get("backups", [])]
        except Exception as e:
            logger.error(f"Failed to load backup manifest: {e}")
            return []

    def _save_manifest(self) -> None:
        """Save backup manifest to disk."""
        try:
            data = {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(),
                "backups": [b.to_dict() for b in self._backups],
            }
            with open(self._manifest_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save backup manifest: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get backup statistics."""
        total_size = sum(b.size_bytes for b in self._backups)
        return {
            "total_backups": len(self._backups),
            "total_size_bytes": total_size,
            "oldest_backup": min(b.created_at for b in self._backups).isoformat() if self._backups else None,
            "newest_backup": max(b.created_at for b in self._backups).isoformat() if self._backups else None,
            "auto_backup_running": self._running,
        }
