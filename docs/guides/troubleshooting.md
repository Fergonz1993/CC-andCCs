# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the Claude Multi-Agent Coordination System.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Option A Issues](#option-a-file-based-issues)
3. [Option B Issues](#option-b-mcp-server-issues)
4. [Option C Issues](#option-c-orchestrator-issues)
5. [Common Errors](#common-errors)
6. [Performance Issues](#performance-issues)
7. [Getting Help](#getting-help)

---

## Quick Diagnostics

Run these commands to quickly diagnose common issues:

```bash
# Check Python version (needs 3.8+)
python3 --version

# Check Node.js version (needs 18+)
node --version

# Check Claude Code is installed
claude --version

# Verify coordination directory structure
ls -la .coordination/

# Check tasks.json is valid JSON
python3 -c "import json; json.load(open('.coordination/tasks.json'))"

# Check file permissions
ls -la .coordination/tasks.json
```

---

## Option A: File-Based Issues

### "No module named 'fcntl'" (Windows)

**Symptom**: Error when running on Windows.

**Cause**: `fcntl` is Unix-only.

**Solution**: The code has a Windows fallback using `msvcrt`. Ensure you're using the latest version:

```python
# coordination.py handles this automatically
if sys.platform == "win32":
    import msvcrt
    # Uses msvcrt.locking instead
```

If you still see errors, the code needs updating to handle Windows properly.

---

### Race Condition: "Task was claimed by another worker"

**Symptom**: Worker claims task but another worker also got it.

**Cause**: Simultaneous claims without proper verification.

**Solution**: Always verify after claiming:

```bash
# The CLI does this automatically, but if using custom code:
# 1. Claim the task
# 2. Re-read tasks.json
# 3. Verify claimed_by matches your ID
# 4. If not, try claiming another task
```

---

### "Permission denied" on tasks.json

**Symptom**: Cannot read or write coordination files.

**Cause**: File permissions or ownership issues.

**Solution**:

```bash
# Fix permissions
chmod 644 .coordination/tasks.json
chmod 755 .coordination/

# Fix ownership
sudo chown -R $(whoami) .coordination/

# Ensure directory is writable
ls -la .coordination/
```

---

### Stale Lock Files

**Symptom**: Operations hang or fail with lock errors.

**Cause**: Lock file left from crashed process.

**Solution**:

```bash
# Remove stale lock files
rm .coordination/*.lock

# If that doesn't work, remove and reinitialize
rm -rf .coordination/
python coordination.py leader init "Your goal"
```

---

### Tasks Not Appearing

**Symptom**: `worker list` shows no tasks but you added some.

**Cause**: Dependencies not satisfied or task status incorrect.

**Solution**:

```bash
# Check the actual task data
cat .coordination/tasks.json | python3 -m json.tool

# Look for:
# - status: should be "available"
# - dependencies: all should be in "done" status
```

---

## Option B: MCP Server Issues

### MCP Server Not Recognized

**Symptom**: Claude Code doesn't show coordination tools.

**Cause**: Configuration issue or server not running.

**Solution**:

```bash
# 1. Verify mcp.json exists and is valid
cat ~/.claude/mcp.json
python3 -c "import json; json.load(open('$HOME/.claude/mcp.json'))"

# 2. Check the path is absolute and correct
ls -la /path/to/option-b-mcp-broker/dist/index.js

# 3. Verify the server can start
node /path/to/option-b-mcp-broker/dist/index.js
# Should output: "Claude Coordination MCP Server running on stdio"

# 4. Restart Claude Code completely
```

---

### "Cannot find module" Error

**Symptom**: Node.js error about missing modules.

**Cause**: Dependencies not installed or build not run.

**Solution**:

```bash
cd option-b-mcp-broker

# Clean and reinstall
rm -rf node_modules dist
npm install
npm run build

# Verify build
ls dist/index.js
```

---

### State Not Persisting

**Symptom**: Tasks disappear after restarting server.

**Cause**: State file not being written or read.

**Solution**:

```bash
# Check state file exists
ls -la .coordination/mcp-state.json

# Verify it has content
cat .coordination/mcp-state.json | python3 -m json.tool

# Check COORDINATION_DIR environment variable
echo $COORDINATION_DIR
```

---

### "Unknown tool" Error

**Symptom**: Tool calls fail with unknown tool error.

**Cause**: Tool name misspelled or server outdated.

**Solution**:

Verify exact tool names:
- `init_coordination`
- `create_task`
- `create_tasks_batch`
- `claim_task`
- `complete_task`
- `fail_task`
- `get_status`
- `get_all_tasks`
- `get_results`
- `register_agent`
- `start_task`
- `heartbeat`
- `add_discovery`
- `get_discoveries`
- `get_master_plan`

---

## Option C: Orchestrator Issues

### "ModuleNotFoundError: No module named 'orchestrator'"

**Symptom**: Python can't find the orchestrator module.

**Cause**: Package not installed or virtual environment not activated.

**Solution**:

```bash
cd option-c-orchestrator

# Activate virtual environment
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Reinstall
pip install -e .

# Verify
python -c "from orchestrator import Orchestrator; print('OK')"
```

---

### "Claude Code CLI not found"

**Symptom**: Orchestrator can't spawn Claude Code processes.

**Cause**: Claude Code not in PATH.

**Solution**:

```bash
# Find Claude Code location
which claude

# If not found, install it
npm install -g @anthropic-ai/claude-code

# Or add to PATH
export PATH="$PATH:/path/to/claude"
```

---

### Tasks Timing Out

**Symptom**: Tasks fail with timeout errors.

**Cause**: Task takes longer than timeout setting.

**Solution**:

```python
# Increase timeout
orch = Orchestrator(
    task_timeout=900  # 15 minutes instead of default 10
)

# Or via CLI
orchestrate run "Goal" --timeout 900
```

---

### Workers Not Claiming Tasks

**Symptom**: Workers idle but tasks available.

**Cause**: Dependencies not satisfied or workers not running.

**Solution**:

```python
# Check task dependencies
for task in orch.state.tasks:
    print(f"{task.id}: deps={task.dependencies}, status={task.status}")

# Check worker status
for agent in orch._pool.get_all_agents():
    print(f"{agent.agent_id}: running={agent.is_running}")
```

---

## Common Errors

### JSON Parsing Errors

**Symptom**: "JSONDecodeError" or "Expecting value"

**Cause**: Corrupted or malformed JSON file.

**Solution**:

```bash
# Validate JSON
python3 -c "import json; json.load(open('.coordination/tasks.json'))"

# If corrupted, check for backups
ls .coordination/*.bak

# Or restore from git
git checkout .coordination/tasks.json

# Last resort: reinitialize
rm .coordination/tasks.json
python coordination.py leader init "Your goal"
```

---

### Circular Dependencies

**Symptom**: Tasks never become available.

**Cause**: Tasks depend on each other in a cycle.

**Solution**:

```bash
# Check for cycles manually
cat .coordination/tasks.json | python3 << 'EOF'
import json
import sys

data = json.load(sys.stdin)
tasks = {t['id']: t.get('dependencies', []) for t in data.get('tasks', [])}

def has_cycle(task_id, visited, rec_stack):
    visited.add(task_id)
    rec_stack.add(task_id)
    for dep in tasks.get(task_id, []):
        if dep not in visited:
            if has_cycle(dep, visited, rec_stack):
                return True
        elif dep in rec_stack:
            print(f"Cycle detected: {task_id} -> {dep}")
            return True
    rec_stack.remove(task_id)
    return False

visited = set()
for tid in tasks:
    if tid not in visited:
        has_cycle(tid, visited, set())
EOF
```

---

### "Task not claimed by this agent"

**Symptom**: Cannot complete or fail a task.

**Cause**: Task was claimed by a different agent or claim failed.

**Solution**:

```bash
# Check who claimed the task
cat .coordination/tasks.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
for t in data['tasks']:
    if t['id'] == 'YOUR_TASK_ID':
        print(f\"Claimed by: {t.get('claimed_by', 'nobody')}\")
"
```

---

## Performance Issues

### Slow File Operations

**Symptom**: Commands take several seconds.

**Cause**: Large tasks.json or slow disk.

**Solution**:

```bash
# Check file size
ls -la .coordination/tasks.json

# Archive completed tasks
python3 << 'EOF'
import json
from datetime import datetime

with open('.coordination/tasks.json', 'r') as f:
    data = json.load(f)

# Separate done tasks
done = [t for t in data['tasks'] if t['status'] == 'done']
active = [t for t in data['tasks'] if t['status'] != 'done']

# Archive done tasks
with open(f'.coordination/archive-{datetime.now().strftime("%Y%m%d")}.json', 'w') as f:
    json.dump({'tasks': done}, f, indent=2)

# Keep only active tasks
data['tasks'] = active
with open('.coordination/tasks.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Archived {len(done)} tasks, kept {len(active)} active")
EOF
```

---

### High Memory Usage (Option C)

**Symptom**: Orchestrator uses excessive memory.

**Cause**: Many concurrent workers or large state.

**Solution**:

```python
# Reduce workers
orch = Orchestrator(max_workers=2)  # Instead of 5

# Process tasks in batches
# Instead of adding all tasks at once
for batch in chunks(all_tasks, 10):
    orch.add_tasks_batch(batch)
    await orch.run_with_predefined_tasks()
```

---

### Slow MCP Response

**Symptom**: MCP tool calls take long to return.

**Cause**: Large state or disk I/O.

**Solution**:

```bash
# Check state file size
ls -la .coordination/mcp-state.json

# Archive old data
# (Backup first!)
cp .coordination/mcp-state.json .coordination/mcp-state.json.bak

# Keep only recent data
python3 << 'EOF'
import json
from datetime import datetime, timedelta

with open('.coordination/mcp-state.json', 'r') as f:
    data = json.load(f)

cutoff = (datetime.now() - timedelta(days=7)).isoformat()

# Filter old tasks
data['tasks'] = [t for t in data.get('tasks', [])
                 if t.get('completed_at', '9999') > cutoff or t['status'] != 'done']

# Filter old discoveries
data['discoveries'] = data.get('discoveries', [])[-100:]  # Keep last 100

with open('.coordination/mcp-state.json', 'w') as f:
    json.dump(data, f, indent=2)
EOF
```

---

## Getting Help

### Collect Diagnostic Information

Before asking for help, collect this information:

```bash
# System info
echo "=== System ===" > diagnostic.txt
uname -a >> diagnostic.txt
python3 --version >> diagnostic.txt
node --version >> diagnostic.txt

# Configuration
echo "=== Config ===" >> diagnostic.txt
cat ~/.claude/mcp.json >> diagnostic.txt 2>/dev/null || echo "No MCP config" >> diagnostic.txt

# Coordination state
echo "=== State ===" >> diagnostic.txt
ls -la .coordination/ >> diagnostic.txt
cat .coordination/tasks.json >> diagnostic.txt 2>/dev/null

# Recent logs
echo "=== Logs ===" >> diagnostic.txt
tail -100 .coordination/logs/*.log >> diagnostic.txt 2>/dev/null
```

### Where to Get Help

1. **Check Documentation**: Review this guide and the FAQ
2. **Search Issues**: Look for similar issues in the project repository
3. **File an Issue**: Include diagnostic information
4. **Community**: Ask in project discussions or forums
