# RALPH_LOOP_PROMPT.md

Run ONE iteration only:
1) Read AGENTS.md (RALPH_STATE), TODO_RALPH.md, README.md/ARCHITECTURE.md.
2) If TODO has unchecked items, pick the next unchecked.
   Else, open feature_list.json and pick the first item where passes != true
   (or, in verification mode, where verified != true).
3) Current target: feature_list:adv-a-003 - Task reassignment when worker goes stale
4) Implement the minimal safe change for that single item.
5) Update TODO_RALPH.md (mark done) or feature_list.json accordingly.
6) Update AGENTS.md with results, pre-mortem, adversarial check, regression net.
7) Stop. Do NOT run tests; the loop will run ./scripts/ralph_loop.py run-tests.
8) If no pending items exist in either list, emit <<<RALPH_DONE>>> and stop.

Verification mode (feature_list.json):
- Re-test or reproduce the feature claim.
- If it holds, add "verified": true plus "verification_date" (YYYY-MM-DD)
  and short "verification_notes".
- If it fails, set "passes": false and add "failure_notes".