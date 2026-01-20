# Developer Setup Guide

This guide walks you through setting up your development environment for working with the Claude Multi-Agent Coordination System.

## Prerequisites

### Required

- **Claude Code**: Install from [claude.ai/code](https://claude.ai/code)
- **Git**: Version control
- **Python 3.8+**: For Options A and C
- **Node.js 18+**: For Option B

### Recommended

- **VS Code** or **Cursor**: IDE with terminal support
- **Multiple terminal windows/tabs**: For running multiple agents

## Quick Verification

```bash
# Verify Claude Code is installed
claude --version

# Verify Python
python3 --version  # Should be 3.8+

# Verify Node.js
node --version  # Should be 18+
npm --version
```

---

## Option A: File-Based Coordination

### Setup

Option A requires only Python standard library - no additional installation needed.

```bash
# Clone the repository
git clone <repository-url>
cd cc-and-ccs/claude-multi-agent/option-a-file-based

# Verify it works
python coordination.py --help
```

### Project Structure

```
option-a-file-based/
├── coordination.py      # Main CLI script
└── README.md           # Option-specific documentation
```

### Development Workflow

1. Open 3+ terminal windows
2. Navigate to your project directory in each
3. Terminal 1: Run leader commands
4. Terminals 2+: Run worker commands

```bash
# Terminal 1 (Leader)
cd /path/to/your/project
python /path/to/coordination.py leader init "Your goal"

# Terminal 2 (Worker)
cd /path/to/your/project
python /path/to/coordination.py worker claim terminal-2
```

### Testing Your Setup

```bash
# Run a quick test
cd /tmp/test-coordination

# Initialize
python /path/to/coordination.py leader init "Test coordination"

# Add a task
python /path/to/coordination.py leader add-task "Test task" -p 1

# Verify structure
ls -la .coordination/
# Should show: master-plan.md, tasks.json, context/, logs/, results/

# Check status
python /path/to/coordination.py leader status
```

---

## Option B: MCP Server

### Installation

```bash
cd cc-and-ccs/claude-multi-agent/option-b-mcp-broker

# Install dependencies
npm install

# Build TypeScript
npm run build

# Verify build succeeded
ls dist/
# Should show: index.js and other compiled files
```

### Configure Claude Code

Edit or create `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "multi-agent-coordinator": {
      "command": "node",
      "args": ["/absolute/path/to/option-b-mcp-broker/dist/index.js"],
      "env": {
        "COORDINATION_DIR": "/path/to/default/working/directory"
      }
    }
  }
}
```

**Important**: Use absolute paths, not relative.

### Restart Claude Code

After configuring, restart all Claude Code instances for the MCP server to be recognized.

### Verify MCP Connection

In Claude Code, you should see the coordination tools available. Try:

```
Use the get_status tool
```

If it returns a status (even if empty), the connection is working.

### Development Mode

For development, use TypeScript directly:

```bash
# Run in development mode (auto-recompile)
npm run dev

# Or watch mode
npm run watch
```

### Project Structure

```
option-b-mcp-broker/
├── src/
│   ├── index.ts       # Main MCP server
│   └── types.ts       # TypeScript types
├── dist/              # Compiled JavaScript
├── package.json       # Dependencies
└── tsconfig.json      # TypeScript config
```

---

## Option C: External Orchestrator

### Installation

```bash
cd cc-and-ccs/claude-multi-agent/option-c-orchestrator

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .

# Verify installation
orchestrate --help
```

### Dependencies

The orchestrator requires:
- `rich`: Terminal UI and formatting
- `pydantic`: Data validation

These are installed automatically with `pip install -e .`

### Project Structure

```
option-c-orchestrator/
├── src/
│   └── orchestrator/
│       ├── __init__.py
│       ├── orchestrator.py   # Main orchestrator class
│       ├── agent.py          # Agent management
│       └── models.py         # Data models
├── examples/                  # Example scripts
├── tests/                     # Test suite
├── pyproject.toml            # Package configuration
└── README.md
```

### Testing Your Setup

```bash
# Run a basic test
cd /tmp/test-orchestrator

orchestrate run "Test goal" -w 1 --timeout 30
```

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=orchestrator
```

---

## IDE Setup

### VS Code

Recommended extensions:
- **Python** (Microsoft)
- **Pylance** (Microsoft)
- **ESLint** (for TypeScript)
- **Prettier** (for TypeScript formatting)

Settings (`.vscode/settings.json`):

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/option-c-orchestrator/.venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.python"
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
```

### Multi-Terminal Setup

For effective development, set up a terminal layout:

```
┌─────────────────────────────────────┐
│  Terminal 1 (Leader)                │
├──────────────────┬──────────────────┤
│  Terminal 2      │  Terminal 3      │
│  (Worker)        │  (Worker)        │
└──────────────────┴──────────────────┘
```

In VS Code, use the split terminal feature (`Ctrl+Shift+5` or `Cmd+Shift+5`).

---

## Environment Variables

### Option A

```bash
export COORDINATION_DIR=".coordination"  # Default location
```

### Option B

```bash
export COORDINATION_DIR="/path/to/project/.coordination"
```

### Option C

```bash
export CLAUDE_MODEL="claude-sonnet-4-20250514"
export MAX_WORKERS="3"
export TASK_TIMEOUT="600"
```

Add to your shell profile (`~/.bashrc`, `~/.zshrc`) for persistence.

---

## Troubleshooting Setup

### Python Issues

```bash
# Wrong Python version
python3 --version  # Should be 3.8+

# Missing pip
python3 -m ensurepip --upgrade

# Virtual environment issues
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Node.js Issues

```bash
# Wrong Node version
node --version  # Should be 18+

# Use nvm to manage versions
nvm install 18
nvm use 18

# Clear npm cache
npm cache clean --force
rm -rf node_modules
npm install
```

### MCP Connection Issues

```bash
# Verify MCP config path
cat ~/.claude/mcp.json

# Check for syntax errors
python3 -c "import json; json.load(open('$HOME/.claude/mcp.json'))"

# Verify server starts
node /path/to/dist/index.js
# Should output: "Claude Coordination MCP Server running on stdio"
```

### File Permission Issues

```bash
# Make scripts executable
chmod +x coordination.py
chmod +x orchestrate

# Fix ownership
sudo chown -R $(whoami) .coordination/
```

---

## Next Steps

Once your environment is set up:

1. **Try the Quickstart**: Follow the [User Guide](user-guide.md) examples
2. **Run Tests**: Verify everything works with test tasks
3. **Read the ADRs**: Understand design decisions in [ADR documentation](../adr/README.md)
4. **Explore Examples**: Check `examples/` directories in each option
