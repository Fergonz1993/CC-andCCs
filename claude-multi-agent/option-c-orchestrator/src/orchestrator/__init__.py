"""
Claude Multi-Agent Orchestrator

A comprehensive system for coordinating multiple Claude Code instances
to work together on complex tasks.

Architecture:
    Orchestrator (this module)
         │
         ├── Leader Agent (Terminal 1)
         │     └── Plans work, creates tasks, aggregates results
         │
         ├── Worker Agent (Terminal 2)
         │     └── Claims and executes tasks
         │
         └── Worker Agent (Terminal 3)
               └── Claims and executes tasks

Communication flows through the orchestrator, which:
1. Spawns Claude Code processes
2. Sends prompts via stdin
3. Receives responses via stdout
4. Coordinates task distribution
5. Aggregates results
"""

from .models import Task, TaskStatus, Agent, AgentRole, CoordinationState
from .agent import ClaudeCodeAgent
from .orchestrator import Orchestrator
from .cli import app

__version__ = "1.0.0"
__all__ = [
    "Task",
    "TaskStatus",
    "Agent",
    "AgentRole",
    "CoordinationState",
    "ClaudeCodeAgent",
    "Orchestrator",
    "app",
]
