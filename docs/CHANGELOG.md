# Changelog

All notable changes to the Claude Multi-Agent Coordination System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- None

### Changed
- None

### Deprecated
- None

### Removed
- None

### Fixed
- None

### Security
- None

---

## [2.1.0] - 2026-01-25

### Added

#### Test Infrastructure (P0)
- ATOM-001: Hypothesis property-based testing for Option C
- ATOM-002: pip-audit for vulnerability scanning in Option C
- ATOM-003: npm audit integration for Option B
- ATOM-004: Fixed ts-jest TS151002 warning with isolatedModules
- ATOM-005: Added .python-version for pyenv consistency
- ATOM-006: Hybrid orchestrator mode test (coordination_dir vs working_directory)
- ATOM-007: File-based orchestrator persistence tests (agents.json, discoveries.json)
- ATOM-008: Explicit error messages for missing coordination files
- ATOM-009: AgentConfig defaults and heartbeat thread teardown tests
- ATOM-010: Smoke test for async orchestrator as CLI default

#### Reliability & Observability (P1)
- ATOM-101: Ralph test gate summary logging to .coordination/logs
- ATOM-102: Coverage reports for pytest and jest
- ATOM-103: Lint/typecheck commands in ralph_config.json
- ATOM-104: Task queue size and throughput metrics export
- ATOM-105: Retry/backoff policy unit tests
- ATOM-106: Dependency enforcement integration tests
- ATOM-107: Duplicate task ID validation in FileOrchestrator
- ATOM-108: Schema validation for tasks.json (detect malformed entries)
- ATOM-109: Structured JSON logging for orchestrator events
- ATOM-110: Performance benchmarks for task claim/complete cycles
- ATOM-111: Agent capability matching tests
- ATOM-112: CLI flag for JSON export of ralph test gate results

#### Developer Experience (P2)
- ATOM-201: .gitignore entries for .codex/ and vercel-agent-skills/
- ATOM-202: Hybrid orchestrator documentation in README
- ATOM-203: Quickstart section for ralph_config.json and ralph_loop.py
- ATOM-204: Sample CI workflow for ralph test gate (.github/workflows/ralph-test-gate.yml)
- ATOM-205: Lint rules for test_doc_generator warnings
- ATOM-206: Pre-commit hook configuration for formatting
- ATOM-207: Property-based testing documentation
- ATOM-208: Dev script for test catalog regeneration
- ATOM-209: Script for cleaning coordination artifacts
- ATOM-210: Changelog template for v2.1 iterations

#### Documentation
- Comprehensive documentation suite including:
  - Architecture Decision Records (ADRs)
  - OpenAPI specification
  - User guide with examples
  - Developer setup guide
  - Troubleshooting guide
  - Performance tuning guide
  - Security best practices guide
  - Migration guide
  - Code examples

### Changed
- Orchestrator now selects backend based on initialization parameters
- pyproject.toml updated with pytest collection filters

### Fixed
- Pytest collection warnings from TestCase/TestModule dataclasses

---

## [1.0.0] - 2024-01-15

### Added

#### Core Features
- Three coordination options:
  - Option A: File-based coordination with Python CLI
  - Option B: MCP Server for real-time coordination
  - Option C: External Python Orchestrator

- Leader-Worker pattern implementation
  - Leader agent for planning and task creation
  - Worker agents for task execution
  - Discovery sharing mechanism

- Task management
  - Task lifecycle (available -> claimed -> in_progress -> done/failed)
  - Priority-based claiming
  - Dependency tracking
  - Context and hints support

#### Option A: File-Based
- `coordination.py` CLI with leader and worker commands
- File locking for race condition prevention
- Polling-based task discovery
- Result aggregation

#### Option B: MCP Server
- TypeScript MCP server implementation
- 15 MCP tools for coordination
- State persistence to JSON
- Real-time task management

#### Option C: Orchestrator
- Python `Orchestrator` class
- Async task execution
- Automatic worker scaling
- Leader-driven planning mode
- Callback support for events

#### Shared Components
- Common task schema (`task.schema.json`)
- Leader and worker prompt templates
- Coordination directory structure

### Documentation
- Main README with quick start guides
- CLAUDE.md for Claude Code integration
- Option-specific READMEs

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2024-01-15 | Initial release |

---

## Upgrade Notes

### Upgrading to 1.0.0

This is the initial release. No upgrade steps required.

### Future Versions

When upgrading between versions:

1. **Backup your state**
   ```bash
   cp -r .coordination .coordination.bak
   ```

2. **Check migration guide**
   See [Migration Guide](guides/migration.md) for version-specific instructions.

3. **Run schema migrations if needed**
   ```bash
   python migrate_schema.py
   ```

4. **Test in a safe environment first**

---

## Changelog Automation

This changelog follows [Keep a Changelog](https://keepachangelog.com/) format.

### Generating Changelog Entries

For contributors, use conventional commits:

```
feat: Add new feature
fix: Fix bug
docs: Update documentation
refactor: Refactor code
test: Add tests
chore: Maintenance
```

### Automation Script

```bash
#!/bin/bash
# generate-changelog.sh
# Generates changelog entry from git commits

VERSION=$1
DATE=$(date +%Y-%m-%d)

echo "## [$VERSION] - $DATE"
echo ""

# Group commits by type
echo "### Added"
git log --oneline --grep="^feat" | sed 's/^[a-f0-9]* /- /'

echo ""
echo "### Fixed"
git log --oneline --grep="^fix" | sed 's/^[a-f0-9]* /- /'

echo ""
echo "### Changed"
git log --oneline --grep="^refactor\|^chore" | sed 's/^[a-f0-9]* /- /'
```

### GitHub Release Notes

When creating a GitHub release:

1. Copy relevant section from this changelog
2. Add any release-specific notes
3. Include download/installation instructions
4. Link to documentation

---

## Links

- [Keep a Changelog](https://keepachangelog.com/)
- [Semantic Versioning](https://semver.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
