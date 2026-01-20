# Video Tutorial Scripts

This directory contains scripts and outlines for video tutorials covering the Claude Multi-Agent Coordination System.

## Tutorial Series Overview

### 1. Getting Started (15 minutes)
Target audience: New users
- [Script: getting-started.md](#getting-started)

### 2. Option A Deep Dive (20 minutes)
Target audience: Users preferring file-based coordination
- [Script: option-a-tutorial.md](#option-a-deep-dive)

### 3. Option B Deep Dive (25 minutes)
Target audience: Users needing real-time coordination
- [Script: option-b-tutorial.md](#option-b-deep-dive)

### 4. Option C Deep Dive (30 minutes)
Target audience: Users needing programmatic control
- [Script: option-c-tutorial.md](#option-c-deep-dive)

### 5. Advanced Workflows (20 minutes)
Target audience: Experienced users
- [Script: advanced-workflows.md](#advanced-workflows)

---

## Getting Started

**Duration**: 15 minutes
**Prerequisites**: Claude Code installed

### Script

```
[INTRO - 0:00-0:30]
Welcome to the Claude Multi-Agent Coordination System tutorial.
In this video, you'll learn how to coordinate multiple Claude Code
instances to work together on development tasks.

[PROBLEM STATEMENT - 0:30-1:30]
Imagine you have a complex project - say, building a full-stack
web application. A single Claude Code instance works sequentially,
but what if you could have multiple instances working in parallel?

That's exactly what this system enables. Think of it as having a
team of AI developers: one leads, others follow and help.

[ARCHITECTURE OVERVIEW - 1:30-3:00]
*Show architecture diagram*

The system has three parts:
1. A Leader agent - plans work and creates tasks
2. Worker agents - claim and execute tasks
3. A coordination layer - how they communicate

We offer three options for that coordination layer:
- Option A: File-based (simplest)
- Option B: MCP Server (real-time)
- Option C: Orchestrator (full automation)

Let's start with Option A since it's the easiest.

[DEMO: OPTION A BASICS - 3:00-8:00]
*Open three terminal windows*

Terminal 1 is our leader. Let's initialize:

$ python coordination.py leader init "Build a simple landing page"

*Show created .coordination directory*

Now let's create some tasks:

$ python coordination.py leader add-task "Create HTML structure" -p 1
$ python coordination.py leader add-task "Add CSS styling" -p 2 -d task-001
$ python coordination.py leader add-task "Add responsive design" -p 3 -d task-002

Notice the dependencies - CSS depends on HTML, responsive depends on CSS.

*Switch to Terminal 2*

In Terminal 2, we're a worker. Let's claim a task:

$ python coordination.py worker claim terminal-2

We got "Create HTML structure" - the only task without dependencies.

*Do some work in Claude Code*

Now let's complete it:

$ python coordination.py worker complete terminal-2 task-001 \
  "Created index.html with header, main, and footer sections" \
  -c index.html

*Switch to Terminal 3, show it can now claim the CSS task*

[WHEN TO USE EACH OPTION - 8:00-10:00]
*Show comparison table*

Use Option A when:
- You want quick setup
- Tasks are simple
- You're learning the system

Use Option B when:
- You need real-time updates
- Multiple users are coordinating
- Production workflows

Use Option C when:
- You need automation
- Complex dependency chains
- CI/CD integration

[NEXT STEPS - 10:00-12:00]
In the next videos, we'll deep dive into each option.

For now, try this yourself:
1. Initialize a coordination session
2. Create 3-4 tasks with dependencies
3. Open another terminal and claim tasks

[TIPS - 12:00-14:00]
A few tips before we go:

1. Right-size your tasks - 5-15 minutes each
2. Be explicit about dependencies
3. Share discoveries - if you find something useful, note it
4. Check status frequently - stay aligned with the team

[OUTRO - 14:00-15:00]
That's it for the getting started guide!
Subscribe for the deep-dive tutorials.
Links in the description.
```

---

## Option A Deep Dive

**Duration**: 20 minutes
**Prerequisites**: Watched Getting Started

### Script Outline

```
[INTRO - 0:00-1:00]
- Welcome back
- Today: Option A in depth
- File-based coordination

[FILE STRUCTURE - 1:00-3:00]
- Walk through .coordination/
- Explain each file's purpose
- Show tasks.json structure

[TASK LIFECYCLE - 3:00-6:00]
- available -> claimed -> in_progress -> done/failed
- Demo each transition
- Show file changes

[RACE CONDITIONS - 6:00-9:00]
- Explain the problem
- Show file locking mechanism
- Demo claim verification
- Handle conflicts

[ADVANCED FEATURES - 9:00-14:00]
- Task context (files, hints)
- Dependencies and ordering
- Agent registration
- Batch operations

[BEST PRACTICES - 14:00-17:00]
- Task design guidelines
- Logging and debugging
- Error handling
- Performance tips

[REAL EXAMPLE - 17:00-19:00]
- Build something real with 2-3 workers
- Show parallel execution
- Aggregate results

[OUTRO - 19:00-20:00]
- Summary
- When to upgrade to Option B
- Next video preview
```

---

## Option B Deep Dive

**Duration**: 25 minutes
**Prerequisites**: Watched Getting Started

### Script Outline

```
[INTRO - 0:00-1:00]
- MCP server coordination
- Real-time, production-ready

[SETUP - 1:00-4:00]
- Install dependencies
- Build TypeScript
- Configure mcp.json
- Verify connection

[MCP TOOLS - 4:00-10:00]
- Leader tools demo
- Worker tools demo
- Shared tools demo
- Resources vs tools

[ARCHITECTURE - 10:00-13:00]
- How MCP works
- State management
- Persistence

[ADVANCED USAGE - 13:00-19:00]
- Batch task creation
- Discovery sharing
- Status monitoring
- Agent management

[PRODUCTION TIPS - 19:00-22:00]
- Security considerations
- Scaling
- Monitoring

[REAL EXAMPLE - 22:00-24:00]
- Multi-user coordination
- Show real-time updates

[OUTRO - 24:00-25:00]
- Summary
- When to use Option C
- Resources
```

---

## Option C Deep Dive

**Duration**: 30 minutes
**Prerequisites**: Watched Getting Started

### Script Outline

```
[INTRO - 0:00-1:00]
- Python orchestrator
- Full programmatic control

[INSTALLATION - 1:00-3:00]
- Virtual environment
- pip install
- Verify

[BASIC USAGE - 3:00-8:00]
- CLI commands
- Python API
- Simple example

[ORCHESTRATOR CLASS - 8:00-14:00]
- Constructor options
- Initialization
- Task management
- Running execution

[LEADER PLANNING - 14:00-18:00]
- Auto-planning mode
- How it works
- Customizing

[CALLBACKS & EVENTS - 18:00-22:00]
- on_task_complete
- on_discovery
- Custom integrations

[CI/CD INTEGRATION - 22:00-26:00]
- GitHub Actions example
- Automated workflows
- Error handling

[REAL EXAMPLE - 26:00-29:00]
- Full automation demo
- Complex dependency graph
- Results aggregation

[OUTRO - 29:00-30:00]
- Summary
- Advanced topics
- Resources
```

---

## Advanced Workflows

**Duration**: 20 minutes
**Prerequisites**: Completed deep dive on at least one option

### Script Outline

```
[INTRO - 0:00-1:00]
- Advanced patterns
- Real-world workflows

[PARALLEL EXECUTION - 1:00-5:00]
- Task graph design
- Maximizing parallelism
- Bottleneck avoidance

[COMPLEX DEPENDENCIES - 5:00-9:00]
- Diamond patterns
- Conditional execution
- Dynamic task creation

[CROSS-OPTION WORKFLOWS - 9:00-13:00]
- Mixing options
- Migration between options
- Hybrid approaches

[ERROR HANDLING - 13:00-16:00]
- Failure recovery
- Retry strategies
- Graceful degradation

[SCALING - 16:00-18:00]
- Many workers
- Large task sets
- Performance tuning

[CASE STUDY - 18:00-19:30]
- Real project walkthrough
- Lessons learned

[OUTRO - 19:30-20:00]
- Summary
- Resources
- Community
```

---

## Production Notes

### Recording Tips

1. **Terminal Setup**
   - Use large, clear font (16pt+)
   - High contrast theme
   - Clean prompt

2. **Screen Layout**
   - 1080p or 4K
   - Clearly visible terminal splits
   - Highlight active terminal

3. **Narration**
   - Speak clearly and slowly
   - Pause before commands
   - Explain what's about to happen

4. **Editing**
   - Cut long waits
   - Add callouts for key points
   - Include chapter markers

### Assets Needed

- Architecture diagrams
- Comparison tables
- Code snippets (syntax highlighted)
- Demo projects
