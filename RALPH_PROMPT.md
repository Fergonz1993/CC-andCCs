# RALPH_PROMPT.md

Run the Codex feature loop:
1) Read AGENTS.md (RALPH_STATE) and TODO_RALPH.md.
2) Pick the next unchecked item (if TODO is empty, check feature_list.json for passes=false).
3) Implement minimal change and run: ./scripts/ralph_loop.py run-tests
4) Update AGENTS.md with results and TODO_RALPH.md with progress (or feature_list.json if used).
5) Repeat until backlog complete, then emit <<<RALPH_DONE>>>.