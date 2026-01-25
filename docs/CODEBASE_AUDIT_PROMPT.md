# CC-and-CCs Codebase Audit Prompt

> **Purpose**: A comprehensive, reusable prompt for deep-dive codebase auditing that systematically surfaces bugs, UX issues, design gaps, backend improvements, and missing features.
>
> **Generated**: 2026-01-22
> **Codebase Stats**: 103 Python files (50,648 LOC) | 49 TypeScript files (23,407 LOC) | 200 features verified

---

## 🎯 Audit Objective

You are performing a comprehensive audit of the **CC-and-CCs** multi-agent coordination system. This system enables multiple Claude Code terminals to work together on complex tasks using a leader-worker pattern.

Your goals:
1. **Understand** all layers of the codebase (CLI → Business Logic → Data → Infrastructure)
2. **Discover** bugs, UX issues, design gaps, backend problems, and missing features
3. **Document** each issue with location, severity, impact, root cause, and proposed solution
4. **Generate** a prioritized implementation roadmap

---

## 📊 Codebase Overview

### Architecture Summary

```
claude-multi-agent/
├── option-a-file-based/     # Python filesystem polling (simplest)
│   └── coordination.py      # ~1500 LOC, CLI-based coordination
├── option-b-mcp-broker/     # TypeScript MCP server (real-time)
│   └── src/index.ts         # ~2500 LOC, real-time via MCP protocol
├── option-c-orchestrator/   # Python async orchestrator (most features)
│   └── src/orchestrator/    # ~3000 LOC, subprocess management
├── shared/                  # Cross-cutting utilities (53 Python files)
│   ├── ai/                  # Task prioritization, decomposition, anomaly detection
│   ├── cross_option/        # Migration, sync, unified CLI, plugins
│   ├── performance/         # Caching, compression, async I/O, profiling
│   ├── reliability/         # Circuit breaker, retry, fallback, self-healing
│   ├── security/            # Auth, encryption, RBAC, audit logging
│   ├── prompts/             # System prompts for leader/worker agents
│   └── schemas/             # task.schema.json
└── tests/integration/       # Cross-option integration tests
```

### Task Lifecycle

```
available → claimed → in_progress → done | failed
```

### Key Components

| Component | File | Role |
|-----------|------|------|
| File Orchestrator | `option-c-orchestrator/src/orchestrator/orchestrator.py` | Sync file-backed coordination with fcntl locking |
| Async Orchestrator | `option-c-orchestrator/src/orchestrator/async_orchestrator.py` | Production async runtime |
| CLI Interface | `option-c-orchestrator/src/orchestrator/cli.py` | Typer-based CLI with rich output |
| Models | `option-c-orchestrator/src/orchestrator/models.py` | Pydantic Task/Agent definitions |
| MCP Server | `option-b-mcp-broker/src/index.ts` | Real-time coordination via MCP |
| File Coordinator | `option-a-file-based/coordination.py` | Filesystem polling approach |
| Task Schema | `shared/schemas/task.schema.json` | Canonical task structure |

---

## 🔍 Audit Checklist

For each area below, examine the code and document issues found:

### 1. Error Handling

- [ ] Are all exceptions caught and handled appropriately?
- [ ] Are error messages user-friendly and actionable?
- [ ] Is graceful degradation implemented for recoverable failures?
- [ ] Are domain-specific exceptions defined (vs generic Exception)?

**Files to examine:**
- `option-c-orchestrator/src/orchestrator/orchestrator.py` (OrchestrationError, TaskSchemaError)
- `option-b-mcp-broker/src/index.ts` (error handlers)
- `option-a-file-based/coordination.py` (try/except patterns)

### 2. Race Conditions & Concurrency

- [ ] Are file operations atomic (write-to-temp + rename pattern)?
- [ ] Is file locking correctly implemented (`fcntl.flock`)?
- [ ] Are TOCTOU vulnerabilities addressed (time-of-check-to-time-of-use)?
- [ ] Is state re-read after acquiring locks?

**Files to examine:**
- `orchestrator.py:79-93` (_file_lock context manager)
- `coordination.py:180-214` (file_lock function)
- `orchestrator.py:337-366` (claim_task with locking)

### 3. Input Validation

- [ ] Are all user inputs sanitized before use?
- [ ] Are path traversal attacks prevented?
- [ ] Is priority range validated (1-10)?
- [ ] Are task IDs validated for format?

**Files to examine:**
- `orchestrator.py:175-242` (_validate_task_schema)
- `cli.py` (typer argument validation)
- `coordination.py` (argparse validation)

### 4. CLI UX/User Experience

- [ ] Are `--help` texts complete and informative?
- [ ] Are error messages actionable (tell user what to do)?
- [ ] Is output formatting consistent (rich panels, tables)?
- [ ] Are interactive prompts available where needed?

**Files to examine:**
- `cli.py` (typer commands)
- `coordination.py:18-32` (CLI docstrings)
- CLI output for status, example, init commands

### 5. Logging & Observability

- [ ] Is logging level-appropriate (debug vs info vs error)?
- [ ] Are sensitive values masked in logs?
- [ ] Are log formats consistent and parseable?
- [ ] Are metrics exposed (Prometheus, JSON)?

**Files to examine:**
- `option-c-orchestrator/src/orchestrator/logging_config.py`
- `option-c-orchestrator/src/orchestrator/metrics.py`
- `coordination.py` (log_action function)

### 6. Test Coverage

- [ ] Are critical paths tested (claim, complete, fail)?
- [ ] Are edge cases covered (empty tasks, invalid IDs)?
- [ ] Is property-based testing used for complex logic?
- [ ] Are integration tests covering cross-option scenarios?

**Files to examine:**
- `option-c-orchestrator/tests/test_orchestrator_unit.py`
- `tests/integration/test_workflow.py`
- `tests/integration/test_benchmarks.py`

### 7. Configuration

- [ ] Are all configuration options documented?
- [ ] Are environment variables validated?
- [ ] Are defaults sensible and documented?
- [ ] Is configuration centralized?

**Files to examine:**
- `option-c-orchestrator/src/orchestrator/config.py`
- `coordination.py` (CONFIGURATION constants)
- `cli.py` (DEFAULT_* imports)

### 8. API Consistency

- [ ] Do similar operations have similar interfaces?
- [ ] Are return types consistent across methods?
- [ ] Is error handling consistent across options?
- [ ] Are naming conventions consistent?

### 9. Security

- [ ] Are file permissions restrictive enough?
- [ ] Is sensitive data (keys, tokens) handled safely?
- [ ] Are dependencies free of known vulnerabilities?
- [ ] Is audit logging implemented?

**Files to examine:**
- `shared/security/` (auth, encryption, audit)
- `option-b-mcp-broker/src/security.ts`
- Run: `pip-audit`, `npm audit`

### 10. Documentation Accuracy

- [ ] Does CLAUDE.md match current implementation?
- [ ] Are code comments up-to-date?
- [ ] Are feature descriptions accurate in feature_list.json?
- [ ] Are CLI examples in docs runnable?

---

## 📝 Issue Documentation Format

For each issue discovered, document it as follows:

```markdown
## Issue: [Category] - [Brief Title]

**ID**: AUDIT-XXX
**Location**: `path/to/file.py:line_range`
**Severity**: Critical | High | Medium | Low
**Category**: Bug | UX/CLI | Design | Backend | Missing Feature
**Impact**: [Who/what is affected]

### Current Behavior
[What happens now - be specific]

### Expected Behavior
[What should happen]

### Root Cause
[Why the issue exists - technical explanation]

### Proposed Solution
[Specific code changes or architectural adjustments]

### Implementation Steps
1. [Step 1 with file path]
2. [Step 2]
3. [Verification step - how to confirm the fix]

### Dependencies
[Other issues that must be fixed first, or blockers]

### Effort Estimate
- Size: Small (1-2 files, <50 LOC) | Medium (3-5 files, <200 LOC) | Large (6+ files)
- Risk: Low | Medium | High
```

---

## 🚀 Roadmap Generation

After documenting issues, organize them into an implementation roadmap:

### Priority Levels

| Priority | Definition | Examples |
|----------|------------|----------|
| **P0 - Critical** | Blocks core functionality, data loss risk | Race conditions, data corruption |
| **P1 - High** | Significant impact on stability/usability | Error handling gaps, CLI confusion |
| **P2 - Medium** | Affects maintainability or scalability | Design inconsistencies, missing tests |
| **P3 - Low** | Nice-to-have improvements | Documentation, code style |

### Implementation Phases

1. **Phase 1: Stability** - Bug fixes that affect data integrity or core workflows
2. **Phase 2: Usability** - CLI improvements and better error messages
3. **Phase 3: Performance** - Backend optimizations and scalability
4. **Phase 4: Maintainability** - Design refactoring and technical debt
5. **Phase 5: Capability** - New features and enhancements

---

## 📋 Expected Output Format

```markdown
# CC-and-CCs Codebase Audit Report

## Executive Summary
- **Total issues found**: X
- **By severity**: Critical: X | High: X | Medium: X | Low: X
- **By category**: Bugs: X | UX: X | Design: X | Backend: X | Missing: X

## Issue Registry

### P0 - Critical
[Issues that block core functionality]

### P1 - High Priority
[Significant improvements to stability/usability]

### P2 - Medium Priority
[Design and architecture enhancements]

### P3 - Low Priority
[Nice-to-have features]

## Implementation Roadmap

### Phase 1: Stability (Weeks 1-2)
- [ ] AUDIT-001: Fix X
- [ ] AUDIT-002: Fix Y

### Phase 2: Usability (Weeks 3-4)
- [ ] AUDIT-003: Improve Z

[etc.]

## Verification Plan
[How to validate each fix was successful]

## Appendix: Files Examined
[List of all files reviewed during audit]
```

---

## 🔧 Diagnostic Commands

Run these before starting the audit:

```bash
# Feature verification status
cat feature_list.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{sum(1 for f in d[\"features\"] if f[\"passes\"])}/{len(d[\"features\"])} passing')"

# Option C import check
cd claude-multi-agent/option-c-orchestrator && source .venv/bin/activate
python -c "from orchestrator import Orchestrator; from orchestrator.models import Task; print('OK')"

# Option B build check
cd claude-multi-agent/option-b-mcp-broker && ls dist/index.js && echo "Build OK"

# Test status
cd claude-multi-agent/option-c-orchestrator && pytest --collect-only -q 2>/dev/null | tail -3
cd claude-multi-agent/option-b-mcp-broker && npm test -- --listTests 2>/dev/null | wc -l

# Security audit
cd claude-multi-agent/option-c-orchestrator && pip-audit 2>/dev/null || echo "pip-audit not installed"
cd claude-multi-agent/option-b-mcp-broker && npm audit 2>/dev/null || echo "npm audit failed"
```

---

## 🎯 Quick Start Audit Commands

To begin the audit, examine these high-priority areas first:

### 1. Concurrency Safety
```bash
# Check file locking patterns
grep -n "fcntl\|LOCK_EX\|LOCK_SH" claude-multi-agent/**/*.py

# Check atomic write patterns
grep -n "\.tmp\|replace\|rename" claude-multi-agent/**/*.py
```

### 2. Error Handling
```bash
# Find exception handling
grep -n "except\|raise\|Error" claude-multi-agent/option-c-orchestrator/src/**/*.py

# Check for bare excepts
grep -n "except:" claude-multi-agent/**/*.py
```

### 3. CLI UX
```bash
# Check help text
cd claude-multi-agent/option-c-orchestrator && orchestrate --help
cd claude-multi-agent/option-a-file-based && python coordination.py --help
```

### 4. Test Coverage
```bash
# Run tests with coverage
cd claude-multi-agent/option-c-orchestrator && pytest --cov=orchestrator --cov-report=term-missing
```

---

## Notes for Auditor

1. **Read before suggesting changes** - Always read the full implementation before proposing fixes
2. **Consider all three options** - Fixes should be consistent across Option A, B, and C
3. **Shared modules matter** - Changes to `shared/` affect all options
4. **Test your fixes** - Every fix should have a verification step
5. **Document dependencies** - Some issues depend on others being fixed first

---

*This prompt can be reused for future audits as the codebase evolves.*
