"""
Option Migration Tool (adv-cross-002)

Provides utilities to migrate coordination state between options:
- A -> B, B -> C, A -> C (forward migrations)
- C -> A, C -> B, B -> A (reverse migrations)

Each migration preserves as much information as possible while
converting to the target option's format.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import shutil

from .task_adapter import (
    UniversalTask,
    TaskAdapter,
    AdapterFactory,
    OptionAAdapter,
    OptionBAdapter,
    OptionCAdapter,
)


class MigrationDirection(Enum):
    """Migration direction enum."""
    A_TO_B = "a_to_b"
    A_TO_C = "a_to_c"
    B_TO_A = "b_to_a"
    B_TO_C = "b_to_c"
    C_TO_A = "c_to_a"
    C_TO_B = "c_to_b"


@dataclass
class MigrationResult:
    """Result of a migration operation."""
    success: bool
    source_option: str
    target_option: str
    tasks_migrated: int
    discoveries_migrated: int
    agents_migrated: int
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    source_path: str = ""
    target_path: str = ""
    backup_path: Optional[str] = None
    migration_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "source_option": self.source_option,
            "target_option": self.target_option,
            "tasks_migrated": self.tasks_migrated,
            "discoveries_migrated": self.discoveries_migrated,
            "agents_migrated": self.agents_migrated,
            "warnings": self.warnings,
            "errors": self.errors,
            "source_path": self.source_path,
            "target_path": self.target_path,
            "backup_path": self.backup_path,
            "migration_time": self.migration_time.isoformat(),
        }


@dataclass
class Discovery:
    """Universal discovery representation."""
    id: str
    agent_id: str
    content: str
    tags: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    related_task: Optional[str] = None


@dataclass
class Agent:
    """Universal agent representation."""
    id: str
    role: str  # 'leader' or 'worker'
    capabilities: List[str] = field(default_factory=list)
    last_seen: Optional[datetime] = None
    tasks_completed: int = 0
    current_task: Optional[str] = None


class MigrationTool:
    """
    Tool for migrating coordination state between options.

    Handles full state migration including:
    - Tasks and their results
    - Discoveries
    - Agent registrations
    - Master plan and goal
    """

    def __init__(self, create_backup: bool = True):
        """
        Initialize the migration tool.

        Args:
            create_backup: Whether to create backups before migration
        """
        self.create_backup = create_backup
        self._source_adapter: Optional[TaskAdapter] = None
        self._target_adapter: Optional[TaskAdapter] = None

    def migrate(
        self,
        source_path: str,
        target_path: str,
        source_option: str,
        target_option: str,
    ) -> MigrationResult:
        """
        Migrate state from one option to another.

        Args:
            source_path: Path to source coordination directory or state file
            target_path: Path to target coordination directory or state file
            source_option: Source option ('A', 'B', or 'C')
            target_option: Target option ('A', 'B', or 'C')

        Returns:
            MigrationResult with migration details
        """
        result = MigrationResult(
            success=False,
            source_option=source_option.upper(),
            target_option=target_option.upper(),
            tasks_migrated=0,
            discoveries_migrated=0,
            agents_migrated=0,
            source_path=source_path,
            target_path=target_path,
        )

        try:
            # Create backup if requested
            if self.create_backup:
                backup_path = self._create_backup(target_path)
                result.backup_path = backup_path

            # Get adapters
            self._source_adapter = AdapterFactory.get_adapter(source_option)
            self._target_adapter = AdapterFactory.get_adapter(target_option)

            # Load source state
            source_state = self._load_state(source_path, source_option)

            # Migrate tasks
            tasks = self._migrate_tasks(source_state)
            result.tasks_migrated = len(tasks)

            # Migrate discoveries
            discoveries = self._migrate_discoveries(source_state)
            result.discoveries_migrated = len(discoveries)

            # Migrate agents
            agents = self._migrate_agents(source_state)
            result.agents_migrated = len(agents)

            # Build target state
            target_state = self._build_target_state(
                source_state,
                tasks,
                discoveries,
                agents,
                target_option,
            )

            # Save target state
            self._save_state(target_state, target_path, target_option)

            result.success = True

        except Exception as e:
            result.errors.append(str(e))

        return result

    def _create_backup(self, path: str) -> str:
        """Create a backup of the target path."""
        target_path = Path(path)
        if not target_path.exists():
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = target_path.parent / f"{target_path.name}.backup_{timestamp}"

        if target_path.is_dir():
            shutil.copytree(target_path, backup_path)
        else:
            shutil.copy2(target_path, backup_path)

        return str(backup_path)

    def _load_state(self, path: str, option: str) -> Dict[str, Any]:
        """Load state from the source option."""
        source_path = Path(path)

        if option.upper() == "A":
            # Option A: Load from .coordination directory
            coord_dir = source_path if source_path.is_dir() else source_path.parent
            tasks_file = coord_dir / "tasks.json"
            discoveries_file = coord_dir / "context" / "discoveries.md"
            agents_file = coord_dir / "agents.json"
            plan_file = coord_dir / "master-plan.md"

            state = {
                "tasks": [],
                "discoveries": [],
                "agents": [],
                "goal": "",
                "master_plan": "",
            }

            if tasks_file.exists():
                with open(tasks_file) as f:
                    data = json.load(f)
                    state["tasks"] = data.get("tasks", [])

            if discoveries_file.exists():
                state["discoveries_md"] = discoveries_file.read_text()

            if agents_file.exists():
                with open(agents_file) as f:
                    data = json.load(f)
                    state["agents"] = data.get("agents", [])

            if plan_file.exists():
                state["master_plan"] = plan_file.read_text()

            return state

        elif option.upper() == "B":
            # Option B: Load from mcp-state.json
            state_file = source_path if source_path.is_file() else source_path / "mcp-state.json"

            if state_file.exists():
                with open(state_file) as f:
                    return json.load(f)

            return {"tasks": [], "discoveries": [], "agents": {}, "goal": "", "master_plan": ""}

        elif option.upper() == "C":
            # Option C: Load from state JSON file
            state_file = source_path if source_path.is_file() else source_path / "state.json"

            if state_file.exists():
                with open(state_file) as f:
                    return json.load(f)

            return {"tasks": [], "discoveries": [], "goal": "", "master_plan": ""}

        else:
            raise ValueError(f"Unknown source option: {option}")

    def _migrate_tasks(self, source_state: Dict[str, Any]) -> List[UniversalTask]:
        """Migrate tasks to universal format."""
        tasks = []
        for task_data in source_state.get("tasks", []):
            try:
                universal_task = self._source_adapter.to_universal(task_data)
                tasks.append(universal_task)
            except Exception:
                # Skip malformed tasks but continue
                pass
        return tasks

    def _migrate_discoveries(self, source_state: Dict[str, Any]) -> List[Discovery]:
        """Migrate discoveries to universal format."""
        discoveries = []

        # Handle list format (Options B and C)
        for disc in source_state.get("discoveries", []):
            if isinstance(disc, dict):
                discovery = Discovery(
                    id=disc.get("id", ""),
                    agent_id=disc.get("agent_id", "unknown"),
                    content=disc.get("content", ""),
                    tags=disc.get("tags", []),
                    created_at=datetime.fromisoformat(disc["created_at"].replace("Z", "+00:00")) if disc.get("created_at") else None,
                    related_task=disc.get("related_task"),
                )
                discoveries.append(discovery)

        # Handle markdown format (Option A)
        if "discoveries_md" in source_state:
            md_content = source_state["discoveries_md"]
            # Parse markdown into discoveries (simple parsing)
            lines = md_content.split("\n")
            current_discovery = []
            for line in lines:
                if line.startswith("## ") or line.startswith("### "):
                    if current_discovery:
                        discoveries.append(Discovery(
                            id=f"md-{len(discoveries)}",
                            agent_id="unknown",
                            content="\n".join(current_discovery),
                        ))
                        current_discovery = []
                current_discovery.append(line)

            if current_discovery:
                discoveries.append(Discovery(
                    id=f"md-{len(discoveries)}",
                    agent_id="unknown",
                    content="\n".join(current_discovery),
                ))

        return discoveries

    def _migrate_agents(self, source_state: Dict[str, Any]) -> List[Agent]:
        """Migrate agents to universal format."""
        agents = []

        # Handle dict format (Option B)
        agents_data = source_state.get("agents", {})
        if isinstance(agents_data, dict):
            for agent_id, agent_info in agents_data.items():
                if isinstance(agent_info, dict):
                    agents.append(Agent(
                        id=agent_info.get("id", agent_id),
                        role=agent_info.get("role", "worker"),
                        last_seen=datetime.fromisoformat(agent_info["last_heartbeat"].replace("Z", "+00:00")) if agent_info.get("last_heartbeat") else None,
                        tasks_completed=agent_info.get("tasks_completed", 0),
                        current_task=agent_info.get("current_task"),
                    ))

        # Handle list format (Options A and C)
        elif isinstance(agents_data, list):
            for agent_info in agents_data:
                if isinstance(agent_info, dict):
                    agents.append(Agent(
                        id=agent_info.get("id", ""),
                        role=agent_info.get("role", "worker"),
                        capabilities=agent_info.get("capabilities", "").split(",") if isinstance(agent_info.get("capabilities"), str) else agent_info.get("capabilities", []),
                        last_seen=datetime.fromisoformat(agent_info["last_seen"]) if agent_info.get("last_seen") else None,
                    ))

        return agents

    def _build_target_state(
        self,
        source_state: Dict[str, Any],
        tasks: List[UniversalTask],
        discoveries: List[Discovery],
        agents: List[Agent],
        target_option: str,
    ) -> Dict[str, Any]:
        """Build target state in the target option's format."""
        if target_option.upper() == "A":
            return {
                "tasks": {
                    "version": "1.0",
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "tasks": [self._target_adapter.from_universal(t) for t in tasks],
                },
                "agents": {
                    "agents": [
                        {
                            "id": a.id,
                            "capabilities": ",".join(a.capabilities) if a.capabilities else "",
                            "registered_at": datetime.now().isoformat(),
                            "last_seen": a.last_seen.isoformat() if a.last_seen else datetime.now().isoformat(),
                        }
                        for a in agents
                    ]
                },
                "discoveries_md": self._format_discoveries_markdown(discoveries),
                "master_plan": source_state.get("master_plan", ""),
                "goal": source_state.get("goal", ""),
            }

        elif target_option.upper() == "B":
            return {
                "master_plan": source_state.get("master_plan", ""),
                "goal": source_state.get("goal", ""),
                "tasks": [self._target_adapter.from_universal(t) for t in tasks],
                "agents": {
                    a.id: {
                        "id": a.id,
                        "role": a.role,
                        "last_heartbeat": a.last_seen.isoformat() if a.last_seen else datetime.now().isoformat(),
                        "current_task": a.current_task,
                        "tasks_completed": a.tasks_completed,
                    }
                    for a in agents
                },
                "discoveries": [
                    {
                        "id": d.id,
                        "agent_id": d.agent_id,
                        "content": d.content,
                        "tags": d.tags,
                        "created_at": d.created_at.isoformat() if d.created_at else datetime.now().isoformat(),
                    }
                    for d in discoveries
                ],
                "created_at": source_state.get("created_at", datetime.now().isoformat()),
                "last_activity": datetime.now().isoformat(),
            }

        elif target_option.upper() == "C":
            return {
                "goal": source_state.get("goal", ""),
                "master_plan": source_state.get("master_plan", ""),
                "tasks": [self._target_adapter.from_universal(t) for t in tasks],
                "discoveries": [
                    {
                        "id": d.id,
                        "agent_id": d.agent_id,
                        "content": d.content,
                        "tags": d.tags,
                        "created_at": d.created_at.isoformat() if d.created_at else datetime.now().isoformat(),
                        "related_task": d.related_task,
                    }
                    for d in discoveries
                ],
                "working_directory": ".",
                "max_parallel_workers": 3,
                "task_timeout_seconds": 600,
                "created_at": source_state.get("created_at", datetime.now().isoformat()),
                "last_activity": datetime.now().isoformat(),
            }

        raise ValueError(f"Unknown target option: {target_option}")

    def _format_discoveries_markdown(self, discoveries: List[Discovery]) -> str:
        """Format discoveries as markdown for Option A."""
        lines = ["# Shared Discoveries", "", "Important findings go here.", ""]

        for disc in discoveries:
            lines.append(f"## Discovery from {disc.agent_id}")
            if disc.created_at:
                lines.append(f"*{disc.created_at.isoformat()}*")
            lines.append("")
            lines.append(disc.content)
            if disc.tags:
                lines.append(f"\nTags: {', '.join(disc.tags)}")
            lines.append("")

        return "\n".join(lines)

    def _save_state(self, state: Dict[str, Any], path: str, option: str) -> None:
        """Save state to the target option's format."""
        target_path = Path(path)

        if option.upper() == "A":
            # Option A: Save to .coordination directory structure
            coord_dir = target_path if target_path.is_dir() or not target_path.suffix else target_path.parent
            coord_dir.mkdir(parents=True, exist_ok=True)
            (coord_dir / "context").mkdir(exist_ok=True)
            (coord_dir / "logs").mkdir(exist_ok=True)
            (coord_dir / "results").mkdir(exist_ok=True)

            with open(coord_dir / "tasks.json", "w") as f:
                json.dump(state["tasks"], f, indent=2)

            with open(coord_dir / "agents.json", "w") as f:
                json.dump(state["agents"], f, indent=2)

            (coord_dir / "context" / "discoveries.md").write_text(state.get("discoveries_md", ""))

            if state.get("master_plan"):
                (coord_dir / "master-plan.md").write_text(state["master_plan"])

        elif option.upper() == "B":
            # Option B: Save to mcp-state.json
            state_file = target_path if target_path.suffix == ".json" else target_path / "mcp-state.json"
            state_file.parent.mkdir(parents=True, exist_ok=True)

            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

        elif option.upper() == "C":
            # Option C: Save to state.json
            state_file = target_path if target_path.suffix == ".json" else target_path / "state.json"
            state_file.parent.mkdir(parents=True, exist_ok=True)

            with open(state_file, "w") as f:
                json.dump(state, f, indent=2, default=str)


# Convenience functions for common migrations

def migrate_a_to_b(source_dir: str, target_file: str) -> MigrationResult:
    """Migrate from Option A to Option B."""
    tool = MigrationTool()
    return tool.migrate(source_dir, target_file, "A", "B")


def migrate_b_to_c(source_file: str, target_file: str) -> MigrationResult:
    """Migrate from Option B to Option C."""
    tool = MigrationTool()
    return tool.migrate(source_file, target_file, "B", "C")


def migrate_a_to_c(source_dir: str, target_file: str) -> MigrationResult:
    """Migrate from Option A to Option C."""
    tool = MigrationTool()
    return tool.migrate(source_dir, target_file, "A", "C")


def migrate_c_to_a(source_file: str, target_dir: str) -> MigrationResult:
    """Migrate from Option C to Option A."""
    tool = MigrationTool()
    return tool.migrate(source_file, target_dir, "C", "A")


def migrate_c_to_b(source_file: str, target_file: str) -> MigrationResult:
    """Migrate from Option C to Option B."""
    tool = MigrationTool()
    return tool.migrate(source_file, target_file, "C", "B")


def migrate_b_to_a(source_file: str, target_dir: str) -> MigrationResult:
    """Migrate from Option B to Option A."""
    tool = MigrationTool()
    return tool.migrate(source_file, target_dir, "B", "A")


def verify_migration(
    source_path: str,
    target_path: str,
    source_option: str,
    target_option: str,
) -> Tuple[bool, List[str]]:
    """
    Verify that a migration was successful.

    Returns:
        Tuple of (success, list of discrepancies)
    """
    source_adapter = AdapterFactory.get_adapter(source_option)
    target_adapter = AdapterFactory.get_adapter(target_option)

    source_tasks = source_adapter.load_tasks(source_path)
    target_tasks = target_adapter.load_tasks(target_path)

    discrepancies = []

    # Check task count
    if len(source_tasks) != len(target_tasks):
        discrepancies.append(
            f"Task count mismatch: source={len(source_tasks)}, target={len(target_tasks)}"
        )

    # Check individual tasks
    source_ids = {t.id for t in source_tasks}
    target_ids = {t.id for t in target_tasks}

    missing_in_target = source_ids - target_ids
    if missing_in_target:
        discrepancies.append(f"Tasks missing in target: {missing_in_target}")

    extra_in_target = target_ids - source_ids
    if extra_in_target:
        discrepancies.append(f"Extra tasks in target: {extra_in_target}")

    # Check task details
    source_by_id = {t.id: t for t in source_tasks}
    target_by_id = {t.id: t for t in target_tasks}

    for task_id in source_ids & target_ids:
        source_task = source_by_id[task_id]
        target_task = target_by_id[task_id]

        if source_task.description != target_task.description:
            discrepancies.append(f"Description mismatch for {task_id}")

        if source_task.status != target_task.status:
            discrepancies.append(f"Status mismatch for {task_id}")

        if source_task.priority != target_task.priority:
            discrepancies.append(f"Priority mismatch for {task_id}")

    return len(discrepancies) == 0, discrepancies
