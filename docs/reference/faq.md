# Frequently Asked Questions (FAQ)

## General Questions

### What is the Claude Multi-Agent Coordination System?

The Claude Multi-Agent Coordination System enables multiple Claude Code instances to work together on complex development tasks. Think of it as having a team of AI developers: one leads (plans and coordinates), others follow (claim and execute tasks).

### Why would I use multiple Claude Code instances?

- **Parallelization**: Multiple instances can work on different tasks simultaneously
- **Specialization**: Different instances can focus on different aspects (frontend, backend, tests)
- **Speed**: Complex projects complete faster with parallel execution
- **Quality**: Each instance can focus deeply on its assigned task

### Which option should I choose?

| Your Situation | Recommended Option |
|----------------|-------------------|
| First time user, quick test | Option A (File-Based) |
| Production use, team environment | Option B (MCP Server) |
| CI/CD automation, scripting | Option C (Orchestrator) |
| Simple tasks, learning | Option A |
| Complex real-time coordination | Option B or C |

### Can I switch between options?

Yes! State can be migrated between options. See the [Migration Guide](../guides/migration.md) for detailed instructions.

---

## Option A: File-Based

### How does file-based coordination work?

All Claude Code instances share the same filesystem. Tasks are stored in a JSON file, and workers poll this file to find available work. File locking prevents race conditions.

### What if two workers claim the same task?

The claim-then-verify protocol prevents this:
1. Worker A claims a task
2. Worker A re-reads the file to verify
3. If another worker (B) claimed it first, Worker A tries a different task

### Why is my worker not seeing new tasks?

Check:
1. Task status is "available"
2. All dependencies are "done"
3. File hasn't been corrupted (validate JSON)

```bash
python3 -c "import json; json.load(open('.coordination/tasks.json'))"
```

### Can I edit tasks.json manually?

Yes, but be careful:
- Maintain valid JSON structure
- Don't change task IDs that have dependencies
- Don't edit while workers are running

---

## Option B: MCP Server

### How do I know if the MCP server is running?

In Claude Code, try using any coordination tool:
```
Use the get_status tool
```

If it returns a response (even empty), the server is working.

### Why aren't my tools showing up?

1. Verify `~/.claude/mcp.json` syntax is valid
2. Check the server path is absolute and correct
3. Restart Claude Code after configuration changes

### Where is state stored?

State is stored in `.coordination/mcp-state.json` in the directory specified by `COORDINATION_DIR`.

### Can multiple projects use the same MCP server?

Yes, but each project should use a different `COORDINATION_DIR`:

```json
{
  "mcpServers": {
    "project-a": {
      "command": "node",
      "args": ["/path/to/dist/index.js"],
      "env": { "COORDINATION_DIR": "/path/to/project-a" }
    },
    "project-b": {
      "command": "node",
      "args": ["/path/to/dist/index.js"],
      "env": { "COORDINATION_DIR": "/path/to/project-b" }
    }
  }
}
```

---

## Option C: Orchestrator

### How does the orchestrator spawn workers?

The orchestrator uses the Claude Code CLI to spawn subprocesses. Each worker is a separate Claude Code instance.

### Can I use the orchestrator without spawning processes?

Yes, you can use the Orchestrator class purely for state management:

```python
orch = Orchestrator()
await orch.initialize("Goal")
orch.add_task("Task 1")
# Manage tasks without running workers
status = orch.get_status()
```

### Why are my tasks timing out?

Default timeout is 600 seconds (10 minutes). For complex tasks:

```python
orch = Orchestrator(task_timeout=1800)  # 30 minutes
```

Or via CLI:
```bash
orchestrate run "Goal" --timeout 1800
```

### Can I run the orchestrator in a Docker container?

Yes, but ensure:
1. Claude Code is installed in the container
2. API credentials are available
3. Working directory is mounted

---

## Tasks and Dependencies

### What's the right task size?

Optimal tasks are 5-15 minutes of work:
- **Too small** (< 2 min): High coordination overhead
- **Too large** (> 20 min): Limits parallelization

### How do I handle task failures?

1. Worker marks task as `failed` with error description
2. Leader reviews failure
3. Options:
   - Fix the issue and reset task to `available`
   - Create a new task with more context
   - Mark as permanently failed

### Can tasks create subtasks?

Workers can suggest subtasks in their results, but only the leader should create actual tasks. This maintains coordination integrity.

### What happens with circular dependencies?

Tasks with circular dependencies will never become available. The system doesn't automatically detect cycles - design your task graph carefully.

---

## Coordination and State

### How do agents share discoveries?

**Option A**: Write to `.coordination/context/discoveries.md`

**Option B**: Use the `add_discovery` tool

**Option C**: Discoveries are captured in `TaskResult` and stored in `state.discoveries`

### Can I resume a coordination session after a crash?

Yes! State is persisted to disk:
- **Option A**: Always on disk (tasks.json)
- **Option B**: Saved after every operation
- **Option C**: Use `save_state()` for checkpoints

### How do I view coordination history?

Check the logs:
```bash
# Option A
cat .coordination/logs/leader.log

# All options - results
ls .coordination/results/
```

---

## Performance

### How many workers should I use?

Depends on your tasks:
- **I/O-bound tasks**: More workers (5-10)
- **CPU-bound tasks**: Fewer workers (2-4)
- **Start with 3**, adjust based on throughput

### Why is coordination slow?

Common causes:
1. **Option A**: Large tasks.json file - archive completed tasks
2. **Option B**: Many discoveries/tasks - clean old data
3. **Option C**: Too many workers - reduce if CPU-bound

### Can I benchmark the system?

Yes, see the benchmark script in examples:
```bash
python docs/examples/scripts/benchmark.py
```

---

## Troubleshooting

### "Permission denied" errors

```bash
# Fix file permissions
chmod 644 .coordination/tasks.json
chmod 755 .coordination/
```

### "JSON decode error"

File is corrupted. Restore from backup:
```bash
# Check for backups
ls .coordination/*.bak

# Or reinitialize
rm .coordination/tasks.json
python coordination.py leader init "Your goal"
```

### Workers stuck in "claimed" status

Worker probably crashed. Reset the task:
```python
import json
with open('.coordination/tasks.json', 'r+') as f:
    data = json.load(f)
    for task in data['tasks']:
        if task['id'] == 'stuck-task-id':
            task['status'] = 'available'
            task['claimed_by'] = None
    f.seek(0)
    json.dump(data, f, indent=2)
    f.truncate()
```

### MCP server "Unknown tool" error

Verify exact tool name spelling. Common mistakes:
- `initCoordination` vs `init_coordination` (use underscore)
- `claimTask` vs `claim_task`

---

## Security

### Is task data encrypted?

By default, no. For sensitive data:
1. Use environment variables for secrets
2. Implement encryption at rest (see Security Guide)
3. Restrict file permissions

### Can I restrict who can create tasks?

**Option B**: Implement role-based access control (see Security Guide)
**Options A/C**: Use file system permissions

---

## Getting Help

### Where can I report bugs?

File an issue on the project repository with:
1. Steps to reproduce
2. Expected vs actual behavior
3. Diagnostic information (see Troubleshooting Guide)

### How can I contribute?

See the [Contributing Guidelines](../CONTRIBUTING.md) for:
- Code contributions
- Documentation improvements
- Bug reports
- Feature requests
