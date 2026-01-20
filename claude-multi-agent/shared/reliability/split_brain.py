"""
Split-Brain Prevention (adv-rel-008)

Implements mechanisms to prevent and detect split-brain scenarios
where multiple nodes think they are the leader.
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable, Set
from enum import Enum

logger = logging.getLogger(__name__)


class FencingState(str, Enum):
    """Fencing states for split-brain prevention."""
    ACTIVE = "active"      # Node is active and participating
    FENCED = "fenced"      # Node is fenced off from cluster
    QUARANTINE = "quarantine"  # Node is in quarantine period


@dataclass
class SplitBrainConfig:
    """Configuration for split-brain prevention."""
    quorum_size: int = 2                  # Minimum nodes for quorum
    fence_timeout: float = 30.0           # Time before fencing a split node
    quarantine_duration: float = 60.0     # Time in quarantine before rejoining
    heartbeat_interval: float = 5.0       # Heartbeat frequency
    fence_file_path: str = ".coordination/fencing"
    use_fencing_tokens: bool = True       # Use fencing tokens for operations


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    id: str
    last_seen: datetime
    state: FencingState = FencingState.ACTIVE
    fencing_token: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FencingToken:
    """Token used to fence operations."""
    token: int
    issued_at: datetime
    issued_by: str
    valid_until: datetime


class SplitBrainPrevention:
    """
    Prevents split-brain scenarios in distributed coordination.

    Mechanisms:
    1. Quorum-based decisions - require majority for leadership
    2. Fencing tokens - prevent stale operations from succeeding
    3. STONITH (Shoot The Other Node In The Head) - fence off split nodes
    4. Epoch/term numbers - detect stale leaders
    """

    def __init__(
        self,
        node_id: str,
        config: Optional[SplitBrainConfig] = None,
        on_fenced: Optional[Callable[[str], None]] = None,
        on_split_detected: Optional[Callable[[List[str]], None]] = None,
    ):
        """
        Initialize split-brain prevention.

        Args:
            node_id: Unique identifier for this node
            config: Configuration options
            on_fenced: Callback when a node is fenced
            on_split_detected: Callback when split-brain is detected
        """
        self.node_id = node_id
        self.config = config or SplitBrainConfig()
        self.on_fenced = on_fenced
        self.on_split_detected = on_split_detected

        self._nodes: Dict[str, NodeInfo] = {}
        self._current_token = 0
        self._state = FencingState.ACTIVE
        self._quarantine_until: Optional[datetime] = None

        self._fence_dir = Path(self.config.fence_file_path)
        self._fence_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Register self
        self._nodes[node_id] = NodeInfo(
            id=node_id,
            last_seen=datetime.now(),
            state=FencingState.ACTIVE,
            fencing_token=0,
        )

    @property
    def is_fenced(self) -> bool:
        """Check if this node is fenced."""
        return self._state == FencingState.FENCED

    @property
    def is_quarantined(self) -> bool:
        """Check if this node is in quarantine."""
        if self._state != FencingState.QUARANTINE:
            return False
        if self._quarantine_until and datetime.now() > self._quarantine_until:
            self._state = FencingState.ACTIVE
            return False
        return True

    @property
    def can_operate(self) -> bool:
        """Check if this node can perform operations."""
        return self._state == FencingState.ACTIVE

    def register_node(self, node_id: str) -> None:
        """Register a new node in the cluster."""
        with self._lock:
            if node_id not in self._nodes:
                self._nodes[node_id] = NodeInfo(
                    id=node_id,
                    last_seen=datetime.now(),
                    fencing_token=self._current_token,
                )
                logger.info(f"Registered node: {node_id}")

    def heartbeat(self, node_id: str) -> None:
        """Update heartbeat for a node."""
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].last_seen = datetime.now()
                if self._nodes[node_id].state == FencingState.QUARANTINE:
                    # Check if quarantine period is over
                    self._nodes[node_id].state = FencingState.ACTIVE

    def has_quorum(self) -> bool:
        """
        Check if we have quorum (majority of nodes are reachable).

        Returns True if enough nodes are active for quorum.
        """
        with self._lock:
            active_count = sum(
                1 for n in self._nodes.values()
                if n.state == FencingState.ACTIVE
                and (datetime.now() - n.last_seen).total_seconds() < self.config.fence_timeout
            )

            total_count = len(self._nodes)
            quorum_needed = (total_count // 2) + 1

            # Also check minimum quorum size
            return active_count >= max(self.config.quorum_size, quorum_needed)

    def get_fencing_token(self) -> FencingToken:
        """
        Get a new fencing token for an operation.

        Fencing tokens ensure that stale operations from a split node
        cannot succeed. Each token has a monotonically increasing number.
        """
        with self._lock:
            if not self.can_operate:
                raise SplitBrainError("Node is fenced or quarantined")

            self._current_token += 1
            token = FencingToken(
                token=self._current_token,
                issued_at=datetime.now(),
                issued_by=self.node_id,
                valid_until=datetime.now() + timedelta(seconds=self.config.fence_timeout),
            )

            # Persist token
            self._save_token(token)

            return token

    def validate_token(self, token: FencingToken) -> bool:
        """
        Validate a fencing token.

        Returns True if the token is valid and can be used.
        """
        with self._lock:
            # Check if token is expired
            if datetime.now() > token.valid_until:
                logger.warning(f"Token {token.token} is expired")
                return False

            # Check if we have a newer token
            if token.token < self._current_token:
                logger.warning(
                    f"Stale token {token.token} (current: {self._current_token})"
                )
                return False

            return True

    def detect_split_brain(self) -> List[List[str]]:
        """
        Detect potential split-brain scenarios.

        Returns list of node groups that appear to be split.
        """
        with self._lock:
            now = datetime.now()
            timeout = timedelta(seconds=self.config.fence_timeout)

            # Group nodes by their last seen time
            active_nodes: List[str] = []
            stale_nodes: List[str] = []

            for node in self._nodes.values():
                if node.state == FencingState.FENCED:
                    continue

                age = now - node.last_seen
                if age > timeout:
                    stale_nodes.append(node.id)
                else:
                    active_nodes.append(node.id)

            # If we have both active and stale nodes, potential split
            if active_nodes and stale_nodes:
                logger.warning(
                    f"Potential split-brain: active={active_nodes}, stale={stale_nodes}"
                )

                if self.on_split_detected:
                    try:
                        self.on_split_detected([active_nodes, stale_nodes])
                    except Exception as e:
                        logger.error(f"on_split_detected callback error: {e}")

                return [active_nodes, stale_nodes]

            return []

    def fence_node(self, node_id: str, reason: str = "") -> bool:
        """
        Fence off a node (STONITH).

        Fenced nodes cannot perform operations until they rejoin.
        """
        with self._lock:
            if node_id not in self._nodes:
                return False

            self._nodes[node_id].state = FencingState.FENCED

            # Write fence file
            fence_file = self._fence_dir / f"{node_id}.fence"
            fence_data = {
                "node_id": node_id,
                "fenced_at": datetime.now().isoformat(),
                "fenced_by": self.node_id,
                "reason": reason,
            }
            fence_file.write_text(json.dumps(fence_data))

            logger.warning(f"Fenced node {node_id}: {reason}")

            if self.on_fenced:
                try:
                    self.on_fenced(node_id)
                except Exception as e:
                    logger.error(f"on_fenced callback error: {e}")

            # If we fenced ourselves, update state
            if node_id == self.node_id:
                self._state = FencingState.FENCED

            return True

    def unfence_node(self, node_id: str) -> bool:
        """
        Remove fencing from a node, putting it in quarantine first.
        """
        with self._lock:
            if node_id not in self._nodes:
                return False

            # Put in quarantine first
            self._nodes[node_id].state = FencingState.QUARANTINE
            self._nodes[node_id].last_seen = datetime.now()

            # Remove fence file
            fence_file = self._fence_dir / f"{node_id}.fence"
            fence_file.unlink(missing_ok=True)

            logger.info(f"Unfenced node {node_id} (now in quarantine)")

            if node_id == self.node_id:
                self._state = FencingState.QUARANTINE
                self._quarantine_until = datetime.now() + timedelta(
                    seconds=self.config.quarantine_duration
                )

            return True

    def check_self_fenced(self) -> bool:
        """Check if this node has been fenced by another node."""
        fence_file = self._fence_dir / f"{self.node_id}.fence"
        if fence_file.exists():
            self._state = FencingState.FENCED
            return True
        return False

    def start_monitoring(self) -> None:
        """Start split-brain monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info("Started split-brain monitoring")

    def stop_monitoring(self) -> None:
        """Stop split-brain monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        logger.info("Stopped split-brain monitoring")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                # Update our heartbeat
                self.heartbeat(self.node_id)

                # Check if we've been fenced
                if self.check_self_fenced():
                    logger.warning(f"Node {self.node_id} has been fenced")

                # Detect split-brain
                splits = self.detect_split_brain()

                # Auto-fence stale nodes if we have quorum
                if splits and self.has_quorum():
                    for stale_node in splits[1]:  # Stale nodes
                        if stale_node != self.node_id:
                            self.fence_node(stale_node, "Stale heartbeat")

            except Exception as e:
                logger.error(f"Split-brain monitor error: {e}")

            time.sleep(self.config.heartbeat_interval)

    def _save_token(self, token: FencingToken) -> None:
        """Save fencing token to disk."""
        token_file = self._fence_dir / "current_token.json"
        data = {
            "token": token.token,
            "issued_at": token.issued_at.isoformat(),
            "issued_by": token.issued_by,
            "valid_until": token.valid_until.isoformat(),
        }
        token_file.write_text(json.dumps(data))

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status for all nodes."""
        with self._lock:
            return {
                "node_id": self.node_id,
                "state": self._state.value,
                "has_quorum": self.has_quorum(),
                "current_token": self._current_token,
                "nodes": {
                    n.id: {
                        "state": n.state.value,
                        "last_seen": n.last_seen.isoformat(),
                        "fencing_token": n.fencing_token,
                    }
                    for n in self._nodes.values()
                },
            }


class SplitBrainError(Exception):
    """Raised when an operation cannot proceed due to split-brain prevention."""
    pass
