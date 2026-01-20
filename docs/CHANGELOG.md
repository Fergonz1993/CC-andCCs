# Changelog

All notable changes to the Claude Multi-Agent Coordination System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation suite
  - Architecture Decision Records (ADRs)
  - OpenAPI specification
  - User guide with examples
  - Developer setup guide
  - Troubleshooting guide
  - Performance tuning guide
  - Security best practices guide
  - Migration guide
  - Code examples
  - Video tutorial scripts
  - FAQ section
  - Glossary of terms
  - Contributing guidelines

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
