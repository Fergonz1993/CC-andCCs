# AGENTS.md

This file provides guidance to OpenAI Codex when working in this repository.

## Ralph Wiggum Loop Instructions

You are operating in a **Ralph Wiggum Loop**. The same prompt is fed to you repeatedly until you complete the task.

### Critical Behaviors

1. **Check Previous Work First**
   - Before doing anything, check what files exist
   - Run `git status` and `git log --oneline -10` to see your previous work
   - Read `.ralph-state.json` for iteration count
   - **DO NOT start from scratch** - build on existing progress

2. **Incremental Progress**
   - Each iteration should move closer to completion
   - If something failed, try a different approach
   - Commit working changes with descriptive messages

3. **Signal Completion**
   - When fully done, output: `<promise>TASK COMPLETE</promise>`
   - Only output this when ALL requirements are verified working
   - Test your changes before claiming completion

### State Files

- `.ralph-state.json` - Current iteration count and status
- `.ralph-logs/` - Logs from previous iterations (read these if stuck)
- `.ralph-last-output.txt` - Your last response

### Common Mistakes to Avoid

- Starting over when files already exist
- Claiming completion without testing
- Ignoring error messages from previous iterations
- Making the same mistake repeatedly

### Git Usage

- Commit after each meaningful change
- Use descriptive commit messages
- Check `git diff` before committing
- Your commits persist between iterations

## Project Context

<!-- Add your project-specific context here -->
