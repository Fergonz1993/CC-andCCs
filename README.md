# Claude Multi-Agent Coordination System

A production-ready system for coordinating multiple Claude Code terminals to work together on complex development tasks. Features a leader-worker pattern where Terminal 1 plans and aggregates work, while Terminals 2+ claim and execute tasks.

## Version 2.1.0 Highlights

- **Hybrid Orchestrator**: Automatic backend selection (file-based for testing, async for production)
- **Property-Based Testing**: Hypothesis-powered test suite for Option C
- **Security Auditing**: pip-audit and npm audit integration
- **Structured Logging**: JSON-formatted orchestrator events with correlation IDs
- **Performance Metrics**: Task queue size and throughput monitoring
- **CI/CD Integration**: GitHub Actions workflow with Ralph test gate

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd claude-multi-agent
./init.sh

# Option A: File-Based (simplest)
cd option-a-file-based
python coordination.py leader init "Build a REST API"
python coordination.py worker claim terminal-2

# Option B: MCP Server
cd option-b-mcp-broker
npm install && npm run build
# Add to ~/.claude/mcp.json, then use via Claude Code

# Option C: Orchestrator (full control)
cd option-c-orchestrator
pip install -e .
orchestrate run "Build user authentication" -w 3
```

## Documentation

- [Full Documentation](claude-multi-agent/README.md) - Complete usage guide
- [Changelog](docs/CHANGELOG.md) - Release history and features
- [Architecture](docs/adr/) - Architecture Decision Records

## Project Structure

```
claude-multi-agent/
  option-a-file-based/   # Python filesystem coordination
  option-b-mcp-broker/   # TypeScript MCP server
  option-c-orchestrator/ # Python orchestrator with CLI
  shared/                # Cross-cutting utilities
  deployment/            # Docker and k8s configs
  tests/                 # Integration tests
```

## License

MIT