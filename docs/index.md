---
layout: default
title: Home
nav_order: 1
---

# Claude Multi-Agent Coordination System

Coordinate multiple Claude Code instances across terminals to work together on complex development tasks. Terminal 1 leads, Terminals 2+ follow and help.

---

## The Problem

Each Claude Code instance runs in isolation with no native inter-process communication. But they share the **same filesystem**, which becomes your coordination layer.

## The Solution

Three coordination options to fit your workflow:

| Option | Best For | Setup Time |
|--------|----------|------------|
| [**A. File-Based**](getting-started.md#option-a-file-based) | Quick prototypes, learning | 5 min |
| [**B. MCP Server**](getting-started.md#option-b-mcp-server) | Production workflows, real-time | 15 min |
| [**C. Orchestrator**](getting-started.md#option-c-external-orchestrator) | Full automation, CI/CD | 10 min |

---

## Quick Example

**Terminal 1 (Leader):**
```bash
cd option-a-file-based
python coordination.py leader init "Build a REST API"
python coordination.py leader add-task "Create User model" -p 1
python coordination.py leader add-task "Create login endpoint" -p 2
```

**Terminal 2 (Worker):**
```bash
python coordination.py worker claim terminal-2
# ... do the work ...
python coordination.py worker complete terminal-2 task-xxx "Done!"
```

**Terminal 3 (Worker):**
```bash
python coordination.py worker claim terminal-3
# ... parallel work ...
```

---

## Documentation

### Getting Started

- [Quick Start Guide](getting-started.md) - Get up and running in 5 minutes
- [User Guide](guides/user-guide.md) - Comprehensive usage guide with examples
- [Developer Setup](guides/developer-setup.md) - Set up your development environment

### API Reference

- [CLI Reference](api/cli-reference.md) - Command-line interface for all options
- [MCP Tools Reference](api/mcp-tools.md) - Option B MCP server tools
- [Python API Reference](api/python-api.md) - Option C programmatic interface

### Architecture

- [System Architecture](architecture/overview.md) - How the system works
- [Architecture Decision Records](adr/README.md) - Key design decisions
- [Glossary](reference/glossary.md) - Terminology and definitions

### Guides

- [Troubleshooting](guides/troubleshooting.md) - Common issues and solutions
- [Performance Tuning](guides/performance-tuning.md) - Optimization strategies
- [Security Best Practices](guides/security.md) - Security guidelines
- [Migration Guide](guides/migration.md) - Moving between options

### Examples & Tutorials

- [Code Examples](examples/README.md) - Working code samples
- [Workflow Examples](examples/workflows.md) - Common patterns
- [Video Tutorial](tutorials/video-storyboard.md) - 5-minute intro walkthrough

---

## Key Concepts

### Leader-Worker Pattern

The system uses a leader-worker pattern:

- **Leader (Terminal 1)**: Plans work, creates tasks, monitors progress, aggregates results
- **Workers (Terminals 2+)**: Claim tasks, execute them, report results

### Task Lifecycle

```
available -> claimed -> in_progress -> done | failed
```

Tasks flow through a defined lifecycle with proper state transitions.

### Coordination Directory

All options use a `.coordination/` directory structure:

```
.coordination/
├── master-plan.md      # Overall goal and approach
├── tasks.json          # Task queue (source of truth)
├── context/
│   └── discoveries.md  # Shared findings
├── logs/               # Per-agent logs
└── results/            # Completed task outputs
```

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| [v2.1.0](releases/v2.1.0.md) | 2026-01-25 | Property-based testing, security scanning, hybrid orchestrator |
| v2.0.0 | 2026-01-20 | 200+ advanced features, AI-driven prioritization |
| v1.0.0 | 2024-01-15 | Initial release with three coordination options |

---

## Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

---

## License

MIT License - see [LICENSE](LICENSE.md) for details.
