#!/usr/bin/env python3
"""
Basic usage example for the Claude Multi-Agent Orchestrator.

This example demonstrates:
1. Creating an orchestrator
2. Initializing with a goal
3. Adding predefined tasks
4. Running the orchestration
5. Processing results
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator import Orchestrator, Task, Discovery


async def main():
    # Configuration
    WORKING_DIR = "."  # Your project directory
    GOAL = "Create a simple REST API with user endpoints"

    # Create orchestrator
    orchestrator = Orchestrator(
        working_directory=WORKING_DIR,
        max_workers=2,  # Start with 2 workers
        model="claude-sonnet-4-20250514",
        task_timeout=300,  # 5 minutes per task
        verbose=True,
    )

    # Callbacks for events
    def on_task_complete(task: Task):
        print(f"\n✓ Task {task.id} completed!")
        if task.result:
            print(f"  Files modified: {task.result.files_modified}")

    def on_discovery(discovery: Discovery):
        print(f"\n💡 Discovery: {discovery.content[:100]}...")

    orchestrator.on_task_complete = on_task_complete
    orchestrator.on_discovery = on_discovery

    # Initialize
    await orchestrator.initialize(GOAL)

    # Add tasks manually (instead of letting leader plan)
    orchestrator.add_task(
        description="Create a User model with id, name, email fields",
        priority=1,
        context_files=["src/models/"],
        hints="Use TypeScript with proper types"
    )

    orchestrator.add_task(
        description="Create GET /users endpoint to list all users",
        priority=2,
        dependencies=[],  # Could depend on model task if we knew its ID
        context_files=["src/routes/"],
    )

    orchestrator.add_task(
        description="Create POST /users endpoint to create a user",
        priority=2,
        context_files=["src/routes/"],
    )

    orchestrator.add_task(
        description="Write basic tests for the user endpoints",
        priority=3,
        hints="Use Jest or Vitest"
    )

    # Run orchestration
    try:
        result = await orchestrator.run_with_predefined_tasks()

        print("\n" + "=" * 60)
        print("ORCHESTRATION COMPLETE")
        print("=" * 60)
        print(f"Tasks completed: {result['tasks_completed']}")
        print(f"Tasks failed: {result['tasks_failed']}")
        print(f"Discoveries: {result['discoveries']}")
        print(f"\nSummary:\n{result['summary']}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
