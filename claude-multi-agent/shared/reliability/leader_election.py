"""
Leader Election for High Availability (adv-rel-007)

Implements leader election mechanisms to ensure high availability
when the primary leader fails.
"""

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from enum import Enum
import json

logger = logging.getLogger(__name__)


class LeaderState(str, Enum):
    """Leader election states."""
    FOLLOWER = "follower"    # Not the leader
    CANDIDATE = "candidate"  # Running for leader
    LEADER = "leader"        # Currently the leader


@dataclass
class LeaderElectionConfig:
    """Configuration for leader election."""
    heartbeat_interval: float = 5.0      # How often leader sends heartbeat
    election_timeout: float = 15.0       # Timeout before starting election
    lease_duration: float = 30.0         # How long leadership lease lasts
    lock_file_path: str = ".coordination/leader.lock"
    state_file_path: str = ".coordination/leader-state.json"


@dataclass
class LeaderInfo:
    """Information about the current leader."""
    id: str
    elected_at: datetime
    last_heartbeat: datetime
    term: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "elected_at": self.elected_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "term": self.term,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LeaderInfo":
        return cls(
            id=data["id"],
            elected_at=datetime.fromisoformat(data["elected_at"]),
            last_heartbeat=datetime.fromisoformat(data["last_heartbeat"]),
            term=data["term"],
            metadata=data.get("metadata", {}),
        )


class LeaderElection:
    """
    Leader election using file-based locking.

    Implements a simple leader election algorithm suitable for
    local multi-agent coordination:

    1. Each candidate tries to acquire a lock file
    2. The holder of the lock is the leader
    3. Leader must periodically renew the lease
    4. If leader fails, another candidate takes over
    """

    def __init__(
        self,
        node_id: str,
        config: Optional[LeaderElectionConfig] = None,
        on_become_leader: Optional[Callable[[], None]] = None,
        on_lose_leadership: Optional[Callable[[], None]] = None,
        on_leader_change: Optional[Callable[[Optional[str], str], None]] = None,
    ):
        """
        Initialize leader election.

        Args:
            node_id: Unique identifier for this node
            config: Election configuration
            on_become_leader: Callback when this node becomes leader
            on_lose_leadership: Callback when this node loses leadership
            on_leader_change: Callback when leader changes (old_id, new_id)
        """
        self.node_id = node_id
        self.config = config or LeaderElectionConfig()
        self.on_become_leader = on_become_leader
        self.on_lose_leadership = on_lose_leadership
        self.on_leader_change = on_leader_change

        self._state = LeaderState.FOLLOWER
        self._term = 0
        self._current_leader: Optional[LeaderInfo] = None
        self._lock_file = Path(self.config.lock_file_path)
        self._state_file = Path(self.config.state_file_path)
        self._lock_fd: Optional[int] = None

        self._running = False
        self._election_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # Ensure directories exist
        self._lock_file.parent.mkdir(parents=True, exist_ok=True)

    @property
    def state(self) -> LeaderState:
        """Get current election state."""
        return self._state

    @property
    def is_leader(self) -> bool:
        """Check if this node is the leader."""
        return self._state == LeaderState.LEADER

    @property
    def current_leader(self) -> Optional[LeaderInfo]:
        """Get current leader info."""
        return self._current_leader

    @property
    def term(self) -> int:
        """Get current term number."""
        return self._term

    def start(self) -> None:
        """Start participating in leader election."""
        if self._running:
            return

        self._running = True
        self._load_state()

        self._election_thread = threading.Thread(
            target=self._election_loop,
            daemon=True,
        )
        self._election_thread.start()

        logger.info(f"Node {self.node_id} started leader election")

    def stop(self) -> None:
        """Stop participating in leader election."""
        self._running = False

        if self._state == LeaderState.LEADER:
            self._release_leadership()

        if self._election_thread:
            self._election_thread.join(timeout=5.0)
            self._election_thread = None

        logger.info(f"Node {self.node_id} stopped leader election")

    def try_become_leader(self) -> bool:
        """
        Attempt to become the leader.

        Returns True if successfully acquired leadership.
        """
        with self._lock:
            if self._state == LeaderState.LEADER:
                return True

            # Check if current leader is still alive
            if self._current_leader:
                age = datetime.now() - self._current_leader.last_heartbeat
                if age.total_seconds() < self.config.election_timeout:
                    # Leader is still alive
                    return False

            # Try to acquire the lock
            return self._acquire_leadership()

    def renew_leadership(self) -> bool:
        """
        Renew the leadership lease.

        Returns True if successfully renewed.
        """
        with self._lock:
            if self._state != LeaderState.LEADER:
                return False

            try:
                self._send_heartbeat()
                return True
            except Exception as e:
                logger.error(f"Failed to renew leadership: {e}")
                self._release_leadership()
                return False

    def step_down(self) -> None:
        """Voluntarily step down from leadership."""
        with self._lock:
            if self._state == LeaderState.LEADER:
                logger.info(f"Node {self.node_id} voluntarily stepping down")
                self._release_leadership()

    def _acquire_leadership(self) -> bool:
        """Try to acquire the leader lock file."""
        try:
            # Try to create lock file exclusively
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

            try:
                self._lock_fd = os.open(str(self._lock_file), flags)
            except FileExistsError:
                # Lock file exists, check if stale
                if self._is_lock_stale():
                    self._lock_file.unlink(missing_ok=True)
                    self._lock_fd = os.open(str(self._lock_file), flags)
                else:
                    return False

            # Write our info to lock file
            self._term += 1
            now = datetime.now()

            self._current_leader = LeaderInfo(
                id=self.node_id,
                elected_at=now,
                last_heartbeat=now,
                term=self._term,
            )

            lock_content = json.dumps(self._current_leader.to_dict())
            os.write(self._lock_fd, lock_content.encode())

            old_state = self._state
            self._state = LeaderState.LEADER

            logger.info(f"Node {self.node_id} became leader (term {self._term})")

            self._save_state()

            if self.on_become_leader:
                try:
                    self.on_become_leader()
                except Exception as e:
                    logger.error(f"on_become_leader callback error: {e}")

            if self.on_leader_change:
                try:
                    self.on_leader_change(None, self.node_id)
                except Exception as e:
                    logger.error(f"on_leader_change callback error: {e}")

            return True

        except Exception as e:
            logger.error(f"Failed to acquire leadership: {e}")
            return False

    def _release_leadership(self) -> None:
        """Release the leader lock."""
        try:
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None

            self._lock_file.unlink(missing_ok=True)

            was_leader = self._state == LeaderState.LEADER
            self._state = LeaderState.FOLLOWER

            if was_leader:
                logger.info(f"Node {self.node_id} released leadership")

                if self.on_lose_leadership:
                    try:
                        self.on_lose_leadership()
                    except Exception as e:
                        logger.error(f"on_lose_leadership callback error: {e}")

        except Exception as e:
            logger.error(f"Error releasing leadership: {e}")

    def _send_heartbeat(self) -> None:
        """Send a heartbeat (update lock file)."""
        if not self._current_leader or self._state != LeaderState.LEADER:
            return

        self._current_leader.last_heartbeat = datetime.now()

        # Update lock file
        try:
            if self._lock_fd is not None:
                os.lseek(self._lock_fd, 0, os.SEEK_SET)
                os.ftruncate(self._lock_fd, 0)
                content = json.dumps(self._current_leader.to_dict())
                os.write(self._lock_fd, content.encode())
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
            raise

        self._save_state()

    def _check_leader_alive(self) -> bool:
        """Check if the current leader is still alive."""
        if not self._lock_file.exists():
            return False

        try:
            content = self._lock_file.read_text()
            leader_data = json.loads(content)
            leader_info = LeaderInfo.from_dict(leader_data)

            age = datetime.now() - leader_info.last_heartbeat

            if age.total_seconds() > self.config.election_timeout:
                return False

            # Update local leader info
            if self._current_leader is None or self._current_leader.id != leader_info.id:
                old_leader = self._current_leader.id if self._current_leader else None
                self._current_leader = leader_info

                if self.on_leader_change and old_leader != leader_info.id:
                    try:
                        self.on_leader_change(old_leader, leader_info.id)
                    except Exception as e:
                        logger.error(f"on_leader_change callback error: {e}")

            return True

        except Exception as e:
            logger.debug(f"Error checking leader: {e}")
            return False

    def _is_lock_stale(self) -> bool:
        """Check if the lock file is stale (leader crashed)."""
        if not self._lock_file.exists():
            return True

        try:
            content = self._lock_file.read_text()
            leader_data = json.loads(content)
            last_heartbeat = datetime.fromisoformat(leader_data["last_heartbeat"])

            age = datetime.now() - last_heartbeat
            return age.total_seconds() > self.config.lease_duration

        except Exception:
            return True

    def _election_loop(self) -> None:
        """Main election loop."""
        while self._running:
            try:
                with self._lock:
                    if self._state == LeaderState.LEADER:
                        # Send heartbeat to maintain leadership
                        try:
                            self._send_heartbeat()
                        except Exception:
                            self._release_leadership()

                    else:
                        # Check if leader is alive
                        if not self._check_leader_alive():
                            # Try to become leader
                            self._state = LeaderState.CANDIDATE
                            self.try_become_leader()

            except Exception as e:
                logger.error(f"Election loop error: {e}")

            time.sleep(self.config.heartbeat_interval)

    def _load_state(self) -> None:
        """Load election state from disk."""
        if self._state_file.exists():
            try:
                data = json.loads(self._state_file.read_text())
                self._term = data.get("term", 0)
            except Exception as e:
                logger.error(f"Failed to load election state: {e}")

    def _save_state(self) -> None:
        """Save election state to disk."""
        try:
            data = {
                "node_id": self.node_id,
                "term": self._term,
                "state": self._state.value,
                "updated_at": datetime.now().isoformat(),
            }
            if self._current_leader:
                data["leader"] = self._current_leader.to_dict()

            self._state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save election state: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get election status."""
        return {
            "node_id": self.node_id,
            "state": self._state.value,
            "is_leader": self.is_leader,
            "term": self._term,
            "current_leader": self._current_leader.to_dict() if self._current_leader else None,
        }
