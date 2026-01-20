#!/usr/bin/env python3
"""
Leader-driven planning example.

This example demonstrates the full workflow where:
1. The leader agent analyzes the goal
2. Creates a task breakdown automatically
3. Workers execute tasks in parallel
4. Leader aggregates results

This is the "hands-off" approach where you just provide a goal
and let the agents figure out how to accomplish it.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator import Orchestrator


async def main():
    # The goal - be as specific as possible for best results
    GOAL = """
    Add a dark mode feature to the application:
    1. Create a theme context/provider
    2. Add a toggle component in the header
    3. Update the CSS to support light/dark themes
    4. Persist the user's preference in localStorage
    5. Add appropriate transitions between themes
    """

    # Create orchestrator with more workers for parallelism
    orchestrator = Orchestrator(
        working_directory=".",
        max_workers=3,
        model="claude-sonnet-4-20250514",
        task_timeout=600,  # 10 minutes - complex tasks need more time
        verbose=True,
    )

    # Initialize with the goal
    await orchestrator.initialize(
        goal=GOAL,
        master_plan="""
        Approach:
        - Analyze existing styling approach (CSS, Tailwind, styled-components?)
        - Create theme system that works with existing code
        - Implement toggle with smooth transitions
        - Ensure accessibility (respect prefers-color-scheme)
        """
    )

    # Run with leader planning
    # The leader will:
    # 1. Analyze the codebase
    # 2. Create appropriate tasks based on what it finds
    # 3. Coordinate workers to execute tasks
    # 4. Aggregate final results
    try:
        result = await orchestrator.run_with_leader_planning()

        # Save results
        orchestrator.save_state("dark-mode-results.json")

        print("\n" + "=" * 60)
        print("FEATURE COMPLETE")
        print("=" * 60)
        print(f"\nSummary:\n{result['summary']}")

        # Show discoveries
        if result['discoveries'] > 0:
            print("\n" + "-" * 60)
            print("KEY DISCOVERIES:")
            for disc in orchestrator.state.discoveries:
                print(f"  • {disc.content}")

    except KeyboardInterrupt:
        print("\nInterrupted - saving partial state...")
        orchestrator.save_state("dark-mode-partial.json")
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
