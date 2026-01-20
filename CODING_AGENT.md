# CODING AGENT - CC-and-CCs Multi-Agent Coordination System

You are continuing work on a long-running autonomous development task.
This is a FRESH context window - you have no memory of previous sessions.

## PROJECT OVERVIEW

**Repository**: CC-and-CCs (Claude Code and Claude Codes coordination)
**Purpose**: Enable multiple Claude Code instances to work together on complex tasks
**Three Coordination Options**:

| Option | Directory | Language | Method |
|--------|-----------|----------|--------|
| A | `option-a-file-based/` | Python | Filesystem polling |
| B | `option-b-mcp-broker/` | TypeScript | MCP server |
| C | `option-c-orchestrator/` | Python | Process management |

---

## STEP 1: GET YOUR BEARINGS (MANDATORY)

Run these commands FIRST before doing anything else:

```bash
# 1. Confirm working directory
pwd

# 2. Read progress from previous sessions
cat claude-progress.txt

# 3. Check feature status
cat feature_list.json | python3 -c "import json,sys; d=json.load(sys.stdin); passing=sum(1 for f in d['features'] if f['passes']); print(f'Features: {passing}/{len(d[\"features\"])} passing')"

# 4. See recent git activity
git log --oneline -5

# 5. Check what's changed
git status
```

---

## STEP 2: SETUP ENVIRONMENT (IF NEEDED)

If this is the first session or setup is incomplete:

```bash
# Make init.sh executable and run it
chmod +x init.sh
./init.sh
```

For individual options:

```bash
# Option A (Python - no setup needed)
cd claude-multi-agent/option-a-file-based
python coordination.py --help

# Option B (TypeScript)
cd claude-multi-agent/option-b-mcp-broker
npm install && npm run build

# Option C (Python with venv)
cd claude-multi-agent/option-c-orchestrator
source .venv/bin/activate
pip install -e .
orchestrate --help
```

---

## STEP 3: VERIFICATION TEST (CRITICAL!)

Before implementing anything NEW, verify existing passing tests still work.
Regressions are UNACCEPTABLE.

```bash
# Check Option C imports
cd claude-multi-agent/option-c-orchestrator
source .venv/bin/activate
python -c "from orchestrator import Orchestrator; from orchestrator.models import Task; print('OK')"

# Check Option B build
cd ../option-b-mcp-broker
ls dist/index.js

# Check Option A CLI
cd ../option-a-file-based
python coordination.py --help
```

If any previously passing test now fails:
1. Mark feature as `"passes": false` in feature_list.json
2. Fix the regression BEFORE any new work
3. Document in claude-progress.txt

---

## STEP 4: CHOOSE A FEATURE TO IMPLEMENT

Look at `feature_list.json` and find a feature with `"passes": false`.

**Priority order** (work on these first):
1. `integration` - End-to-end tests (proves system works)
2. `option-c-models` - Foundation models (everything depends on these)
3. `option-c-cli` - User interface
4. `option-c-agent` - Process management
5. `option-c-orchestrator` - Coordination logic
6. `option-a-*` - File-based option
7. `option-b-*` - MCP option

**Strategy**: Pick ONE feature. Complete it fully before moving on.

---

## STEP 5: IMPLEMENT THE FEATURE

For each feature, the `steps` array tells you exactly what to verify.

Example feature:
```json
{
  "id": "opt-c-model-001",
  "description": "Task model generates ID with task- prefix",
  "steps": [
    "python -c \"from orchestrator.models import Task; t = Task(description='Test'); print(t.id)\"",
    "Verify ID starts with 'task-'"
  ],
  "passes": false
}
```

**Your job**:
1. Run the steps exactly as written
2. If it fails, fix the code
3. If it passes, mark `"passes": true`

---

## STEP 6: UPDATE feature_list.json

You can ONLY modify the `"passes"` field.

**NEVER**:
- Remove features
- Edit descriptions
- Change steps
- Add new features

**ONLY**:
```json
"passes": true   // After thorough verification
```

---

## STEP 7: UPDATE claude-progress.txt

Add your session entry at the top of the Session Log:

```
[2026-01-19 22:00] SESSION #1 - Verified Option C models
- Tested features: opt-c-model-001 through opt-c-model-010
- All 10 passed, marked in feature_list.json
- No bugs found
- Next: Continue with opt-c-model-011
```

Update the counts at the top:
```
- Option C (Orchestrator): 10/105 passing
Overall Progress: 10/210 (5%)
```

---

## STEP 8: COMMIT YOUR PROGRESS

```bash
git add .
git commit -m "Verify [feature-id]: [description]

- [specific changes made]
- Progress: X/210 features passing

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

Commit frequently - at least after each feature verification.

---

## STEP 9: END SESSION CLEANLY

Before your context fills up:

1. **Commit all code** - Don't leave uncommitted changes
2. **Update claude-progress.txt** - Include session summary
3. **Update feature_list.json** - Mark verified features
4. **Leave system working** - Don't break things mid-change

---

## KEY FILE REFERENCE

| File | Purpose |
|------|---------|
| `feature_list.json` | 210 features to verify |
| `claude-progress.txt` | Session history and notes |
| `init.sh` | Environment setup |
| `CLAUDE.md` | Project overview |
| `CODING_AGENT.md` | This file (session instructions) |

### Option A Files
| File | Purpose |
|------|---------|
| `option-a-file-based/coordination.py` | Main CLI |
| `option-a-file-based/.coordination/` | Coordination directory |

### Option B Files
| File | Purpose |
|------|---------|
| `option-b-mcp-broker/src/index.ts` | MCP server |
| `option-b-mcp-broker/dist/index.js` | Compiled output |
| `option-b-mcp-broker/package.json` | Dependencies |

### Option C Files
| File | Purpose |
|------|---------|
| `option-c-orchestrator/src/orchestrator/models.py` | Pydantic models |
| `option-c-orchestrator/src/orchestrator/agent.py` | Claude subprocess |
| `option-c-orchestrator/src/orchestrator/orchestrator.py` | Main coordinator |
| `option-c-orchestrator/src/orchestrator/cli.py` | Typer CLI |
| `option-c-orchestrator/pyproject.toml` | Python config |

---

## KNOWN ISSUES TO FIX

### 1. Agent Communication (CRITICAL)
**Location**: `option-c-orchestrator/src/orchestrator/agent.py`
**Problem**: Uses `--print` flag which exits after one prompt
**Impact**: Workers can't execute multiple tasks sequentially
**Solution options**:
- Spawn new process for each task
- Use Claude SDK directly instead of CLI
- Use `--conversation` mode with careful stdin/stdout management

### 2. Dependency Cycles
**Location**: `models.py` and `orchestrator.py`
**Problem**: No validation for circular dependencies
**Impact**: `add_task()` could create deadlocks
**Solution**: Add topological sort validation before accepting dependencies

### 3. Worker Recovery
**Location**: `orchestrator.py`
**Problem**: No heartbeat timeout handling
**Impact**: Crashed workers leave tasks stuck in `in_progress`
**Solution**: Add timeout detection and task reset logic

---

## TESTING PATTERNS

### Testing Python (Option C)
```bash
cd claude-multi-agent/option-c-orchestrator
source .venv/bin/activate

# Quick import test
python -c "from orchestrator.models import Task; print(Task(description='test').id)"

# Interactive testing
python
>>> from orchestrator import Orchestrator
>>> from orchestrator.models import Task, TaskStatus
>>> t = Task(description="Test task")
>>> t.claim("agent-1")
>>> print(t.status)
```

### Testing TypeScript (Option B)
```bash
cd claude-multi-agent/option-b-mcp-broker

# Build and check
npm run build
node -e "console.log('Build OK')"

# Check compiled output exists
ls -la dist/
```

### Testing CLI (Option A)
```bash
cd claude-multi-agent/option-a-file-based

# Full workflow test
rm -rf .coordination
python coordination.py leader init "Test goal"
python coordination.py leader add-task "Task 1" -p 1
python coordination.py worker claim terminal-1
python coordination.py leader status
```

---

## REMEMBER

1. **Goal**: Get all 210 features passing
2. **This session**: Complete at least one feature perfectly
3. **Priority**: Fix regressions before new work
4. **Quality**: Zero errors, clean code
5. **Documentation**: Update progress files
6. **Commits**: Frequent, with clear messages

Your work persists through `feature_list.json` and `claude-progress.txt`.
Future sessions depend on the quality of your updates.

Good luck!
