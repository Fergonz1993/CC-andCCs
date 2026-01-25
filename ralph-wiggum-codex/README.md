# Ralph Wiggum Loop for OpenAI Codex CLI

> "I'm in danger... of iterating forever" - Ralph Wiggum

A working implementation of the [Ralph Wiggum technique](https://ghuntley.com/ralph/) for OpenAI's Codex CLI.

## Prerequisites

1. **ChatGPT Plus/Pro/Team/Enterprise subscription** (Codex CLI requires this)
2. **Node.js 18+**

## Setup

```bash
# 1. Install Codex CLI
npm i -g @openai/codex

# 2. Sign in (opens browser)
codex

# 3. Clone this setup to your project
cp -r ralph-wiggum-codex/ /path/to/your/project/
cd /path/to/your/project/

# 4. Initialize git (required for Codex)
git init  # if not already a git repo
```

## Usage

### Basic

```bash
./ralph-loop.sh "Add unit tests for the auth module"
```

### With Options

```bash
./ralph-loop.sh "Fix the login bug and add error handling" \
  --max-iterations 20 \
  --completion-promise "BUG FIXED" \
  --full-auto
```

### Options

| Flag | Description |
|------|-------------|
| `--max-iterations N` | Stop after N iterations (default: 50) |
| `--completion-promise TEXT` | Text that signals completion (default: "TASK COMPLETE") |
| `--working-dir DIR` | Run in different directory |
| `--full-auto` | Auto-approve safe operations |
| `--yolo` | **DANGEROUS**: No sandbox, no approvals |

## How It Works

```
┌──────────────────────────────────────────────────────┐
│  1. Your prompt + Ralph instructions → codex exec    │
│  2. Codex modifies files, runs commands              │
│  3. Check output for <promise>DONE</promise>         │
│  4. Found? → EXIT SUCCESS                            │
│  5. Not found? → LOOP (same prompt again)            │
│  6. Max iterations? → EXIT FAILURE                   │
└──────────────────────────────────────────────────────┘
```

**Key insight:** Codex doesn't remember previous iterations. It sees its work in:
- Modified files in the filesystem
- Git history (commits it made)
- `.ralph-state.json` (iteration counter)
- `.ralph-logs/` (previous output logs)

## Files

| File | Purpose |
|------|---------|
| `ralph-loop.sh` | Main loop script |
| `cancel-ralph.sh` | Stop an active loop |
| `AGENTS.md` | Instructions for Codex (like CLAUDE.md) |
| `.ralph-state.json` | Loop state (created at runtime) |
| `.ralph-logs/` | Iteration logs (created at runtime) |

## Writing Good Tasks

### Good Task

```bash
./ralph-loop.sh "Add JWT authentication to /api/auth endpoint.
Create login and refresh token routes.
Add middleware for protected routes.
Run 'npm test' to verify.
Output <promise>AUTH COMPLETE</promise> when all tests pass."
```

### Bad Task

```bash
./ralph-loop.sh "Make auth better"  # Too vague
```

### Tips

1. **Be specific** - Include exact file names, endpoints, behaviors
2. **Include verification** - Tell it how to test ("run npm test")
3. **Define done** - Clear success criteria
4. **Mention the promise** - Remind it to output `<promise>TEXT</promise>`

## Example Session

```bash
$ ./ralph-loop.sh "Create a /health endpoint that returns {status: 'ok'}" --max-iterations 10

╔═══════════════════════════════════════════════════════════╗
║         RALPH WIGGUM LOOP - CODEX CLI EDITION             ║
╚═══════════════════════════════════════════════════════════╝

[Config] Max iterations: 10
[Config] Completion promise: 'TASK COMPLETE'

[Ralph] Iteration 1 / 10
[Ralph] Executing codex...
... codex creates the endpoint ...

[Ralph] No completion promise detected. Continuing loop...

[Ralph] Iteration 2 / 10
[Ralph] Executing codex...
... codex tests and outputs <promise>TASK COMPLETE</promise> ...

╔═══════════════════════════════════════════════════════════╗
║                    TASK COMPLETE!                         ║
╚═══════════════════════════════════════════════════════════╝

[Ralph] Completed in 2 iterations
```

## Troubleshooting

### "codex: command not found"

```bash
npm i -g @openai/codex
```

### "Not authenticated"

```bash
codex  # Run interactively to sign in
```

### Loop never completes

- Check `.ralph-logs/` to see what Codex is doing
- Make sure your completion promise matches exactly
- Add clearer success criteria to your prompt

### Codex keeps starting over

- Ensure you're in a git repo (`git init`)
- Check that AGENTS.md exists and has the instructions

## Canceling a Loop

```bash
# Ctrl+C during execution, or:
./cancel-ralph.sh
```

## Sources

- [OpenAI Codex CLI](https://github.com/openai/codex)
- [Codex Documentation](https://developers.openai.com/codex/cli/)
- [Original Ralph Wiggum Technique](https://ghuntley.com/ralph/)

## License

MIT
