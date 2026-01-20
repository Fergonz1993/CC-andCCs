# Claude Multi-Agent Orchestrator

External orchestrator for coordinating multiple Claude Code instances.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Let leader plan automatically
orchestrate run "Build a user authentication system" -w 3

# With predefined tasks
orchestrate run "Build API" --no-plan --tasks tasks.json
```

See main README.md for full documentation.
