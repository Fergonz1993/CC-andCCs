---
layout: default
title: Video Tutorial Storyboard
parent: Tutorials
nav_order: 1
---

# Video Tutorial: Getting Started with Multi-Agent Coordination

**Duration:** 5 minutes
**Target Audience:** Developers new to Claude Code multi-agent workflows
**Objective:** Show how to coordinate multiple Claude Code instances to complete a real task

---

## Video Overview

This tutorial demonstrates how to use the Claude Multi-Agent Coordination System to parallelize development work across multiple Claude Code terminals.

### Prerequisites for Viewers

- Claude Code installed (`npm install -g @anthropic-ai/claude-code`)
- Python 3.8+ installed
- Two or more terminal windows available
- Basic familiarity with command line

---

## Section 1: Introduction (0:00 - 0:45)

### Script

> "Welcome to the Claude Multi-Agent Coordination System tutorial. If you've ever wished you could have multiple AI assistants working on different parts of your project simultaneously, this is for you.
>
> Today we'll set up a leader-worker coordination system where Terminal 1 plans the work, and Terminals 2 and 3 execute tasks in parallel.
>
> By the end of this video, you'll have a working multi-agent setup that can tackle complex development tasks faster than ever."

### Screen Recording Instructions

1. **Opening shot**: Show a split terminal view with 3 terminal panes
2. **Overlay text**: "Claude Multi-Agent Coordination" with version number
3. **Brief diagram**: Show the leader-worker pattern (Leader -> Workers)

### Visual Elements

- Title card: "Multi-Agent Coordination with Claude Code"
- Animated diagram showing Terminal 1 (blue) connecting to Terminals 2, 3 (orange)
- Text overlay: "5-Minute Quick Start"

---

## Section 2: Setup (0:45 - 1:30)

### Script

> "Let's start by setting up the coordination system. We'll use Option A, the file-based approach, which requires no additional installation.
>
> First, clone the repository and navigate to the file-based option. This method uses your filesystem as the communication layer between terminals - simple but effective."

### Screen Recording Instructions

1. **Terminal 1**: Run the following commands:
   ```bash
   git clone https://github.com/your-org/claude-multi-agent.git
   cd claude-multi-agent/option-a-file-based
   ls -la
   ```

2. **Show output**: Highlight `coordination.py` file

3. **Quick help check**:
   ```bash
   python coordination.py --help
   ```

4. **Show the help output** briefly (2-3 seconds)

### Visual Elements

- Terminal with commands highlighted
- Callout box: "No installation needed - uses Python standard library"
- File tree showing project structure

---

## Section 3: First Coordination (1:30 - 3:30)

### Script

> "Now let's create our first coordinated task. Imagine we want to build a simple REST API. I'll set up the leader in Terminal 1, then show how workers in Terminals 2 and 3 pick up and execute tasks.
>
> In Terminal 1, I'll initialize the coordination and create three tasks. Notice how I set priorities - lower numbers mean higher priority. I also add dependencies so tasks execute in the right order."

### Demo Scenario

**Goal**: "Build a simple user API with tests"

**Tasks**:
1. Create User model (priority 1)
2. Create GET /users endpoint (priority 2, depends on task 1)
3. Write unit tests (priority 3, depends on task 2)

### Screen Recording Instructions

**Terminal 1 (Leader):**

```bash
# Initialize coordination
python coordination.py leader init "Build a simple user API with tests"

# Add tasks
python coordination.py leader add-task "Create User model with id, name, email fields" -p 1

python coordination.py leader add-task "Create GET /users endpoint that returns all users" -p 2 -d task-001

python coordination.py leader add-task "Write unit tests for User model and endpoint" -p 3 -d task-002

# Check status
python coordination.py leader status
```

**Pause and explain**: Show the status output, highlight the task queue

**Terminal 2 (Worker):**

```bash
cd claude-multi-agent/option-a-file-based

# Claim a task
python coordination.py worker claim terminal-2

# Show what was claimed
# (The output shows task-001: Create User model)
```

**Pause**: Explain that the worker got the highest priority available task

**Simulate work completion:**

```bash
# After doing the work...
python coordination.py worker complete terminal-2 task-001 \
  "Created User model with TypeScript interface" \
  -c src/models/user.ts
```

**Terminal 3 (Worker):**

```bash
cd claude-multi-agent/option-a-file-based

# Claim a task (will get task-002 since task-001 is done)
python coordination.py worker claim terminal-3
```

### Visual Elements

- Split screen showing all 3 terminals
- Animated arrows showing task flow
- Progress bar updating as tasks complete
- Callout: "Dependencies ensure correct execution order"

---

## Section 4: Monitoring Progress (3:30 - 4:15)

### Script

> "Back in Terminal 1, let's check our progress. The leader status command shows all tasks grouped by status.
>
> Notice how task 2 became available only after task 1 completed - that's the dependency system at work.
>
> You can also view the results of completed tasks and aggregate everything at the end."

### Screen Recording Instructions

**Terminal 1:**

```bash
# Check status
python coordination.py leader status

# Show results
cat .coordination/results/task-001.md

# Show the coordination directory structure
ls -la .coordination/
```

**Show the directory structure:**
```
.coordination/
├── master-plan.md
├── tasks.json
├── context/
│   └── discoveries.md
├── logs/
│   ├── leader.log
│   └── terminal-2.log
└── results/
    └── task-001.md
```

### Visual Elements

- Status output with color-coded task states
- File tree animation showing coordination directory
- Highlight the results folder

---

## Section 5: Conclusion (4:15 - 5:00)

### Script

> "That's the basics of multi-agent coordination! You've seen how to:
>
> 1. Initialize a coordination session
> 2. Create tasks with priorities and dependencies
> 3. Have workers claim and complete tasks
> 4. Monitor overall progress
>
> This was Option A, the file-based approach. For real-time coordination, check out Option B which uses the MCP protocol. For full automation, Option C provides a Python orchestrator that can spawn and manage workers automatically.
>
> Links to all documentation are in the description. Happy coordinating!"

### Screen Recording Instructions

1. **Final status check** showing completed tasks
2. **Aggregate results**:
   ```bash
   python coordination.py leader aggregate
   cat .coordination/summary.md
   ```

3. **Show the three options briefly** (quick text overlay or slide)

### Visual Elements

- Summary slide with 3 options comparison
- Links to documentation
- "Subscribe and like" call to action
- End card with related videos

---

## Demo Scenarios

### Scenario A: REST API (Used in Main Tutorial)

**Goal**: Build a simple user API
**Tasks**:
1. Create User model
2. Create endpoints
3. Write tests

### Scenario B: Bug Fix (Alternative)

**Goal**: Fix performance issue in search endpoint
**Tasks**:
1. Profile current performance
2. Identify bottlenecks
3. Implement fixes
4. Verify improvements

### Scenario C: Feature Addition (Alternative)

**Goal**: Add user authentication
**Tasks**:
1. Create JWT utilities
2. Implement login endpoint
3. Add auth middleware
4. Protect existing routes

---

## Screen Recording Checklist

### Before Recording

- [ ] Clean terminal history (`history -c`)
- [ ] Increase terminal font size (18pt recommended)
- [ ] Remove personal information from prompt
- [ ] Close unnecessary applications
- [ ] Silence notifications

### Terminal Setup

- [ ] 3 terminal panes visible (split view)
- [ ] Each terminal labeled (Terminal 1, 2, 3)
- [ ] Dark theme recommended for visibility
- [ ] 1920x1080 resolution minimum

### Audio Setup

- [ ] Quiet recording environment
- [ ] Microphone test completed
- [ ] Script reviewed and practiced

### Post-Production

- [ ] Add intro/outro cards
- [ ] Add subtitles/captions
- [ ] Include chapter markers
- [ ] Add links to description

---

## Timing Breakdown

| Section | Start | End | Duration |
|---------|-------|-----|----------|
| Introduction | 0:00 | 0:45 | 45 sec |
| Setup | 0:45 | 1:30 | 45 sec |
| First Coordination | 1:30 | 3:30 | 2 min |
| Monitoring Progress | 3:30 | 4:15 | 45 sec |
| Conclusion | 4:15 | 5:00 | 45 sec |

---

## Additional Resources

### Link in Description

```
Documentation: https://your-org.github.io/claude-multi-agent/
GitHub: https://github.com/your-org/claude-multi-agent
Getting Started: https://your-org.github.io/claude-multi-agent/getting-started
API Reference: https://your-org.github.io/claude-multi-agent/api/cli-reference
```

### Related Video Ideas

1. "Deep Dive: Option B MCP Server Setup"
2. "Option C: Fully Automated Orchestration"
3. "Advanced: Complex Dependency Graphs"
4. "CI/CD Integration with Multi-Agent Coordination"

---

## Script Full Text

[Full narration script for teleprompter or voice-over]

```
Welcome to the Claude Multi-Agent Coordination System tutorial. If you've ever wished you could have multiple AI assistants working on different parts of your project simultaneously, this is for you.

Today we'll set up a leader-worker coordination system where Terminal 1 plans the work, and Terminals 2 and 3 execute tasks in parallel.

By the end of this video, you'll have a working multi-agent setup that can tackle complex development tasks faster than ever.

[SETUP]

Let's start by setting up the coordination system. We'll use Option A, the file-based approach, which requires no additional installation.

First, clone the repository and navigate to the file-based option. This method uses your filesystem as the communication layer between terminals - simple but effective.

[FIRST COORDINATION]

Now let's create our first coordinated task. Imagine we want to build a simple REST API. I'll set up the leader in Terminal 1, then show how workers in Terminals 2 and 3 pick up and execute tasks.

In Terminal 1, I'll initialize the coordination and create three tasks. Notice how I set priorities - lower numbers mean higher priority. I also add dependencies so tasks execute in the right order.

Watch as Terminal 2 claims the first task. Since it has priority 1 and no dependencies, it's available immediately.

Now I'll complete this task and show how Terminal 3 can then claim the next available task.

[MONITORING]

Back in Terminal 1, let's check our progress. The leader status command shows all tasks grouped by status.

Notice how task 2 became available only after task 1 completed - that's the dependency system at work.

You can also view the results of completed tasks and aggregate everything at the end.

[CONCLUSION]

That's the basics of multi-agent coordination! You've seen how to initialize a coordination session, create tasks with priorities and dependencies, have workers claim and complete tasks, and monitor overall progress.

This was Option A, the file-based approach. For real-time coordination, check out Option B which uses the MCP protocol. For full automation, Option C provides a Python orchestrator that can spawn and manage workers automatically.

Links to all documentation are in the description. Happy coordinating!
```
