# TODO_RALPH.md

End goal: Expand the multi-agent coordination system with a hardened test gate, clearer hybrid orchestrator boundaries, and broader reliability/observability coverage for v2.1.
Status: v2.1 backlog complete. New 200-feature list lives in feature_list.json (api/observability/deployment pending verification).

## P0 - Correctness / Security / Critical Path
- [x] ATOM-001: Add Hypothesis to Option C dev deps and make property-based tests run in CI
- [x] ATOM-002: Add pip-audit to Option C dev deps and wire it into ralph test gate
- [x] ATOM-003: Add npm audit --omit=dev to ralph test gate for Option B and core extensions
- [x] ATOM-004: Resolve ts-jest TS151002 warning by setting isolatedModules or diagnostics ignore
- [x] ATOM-005: Add .python-version aligned with Option C venv to avoid pyenv drift
- [x] ATOM-006: Add hybrid orchestrator mode test that verifies coordination_dir vs working_directory selection
- [x] ATOM-007: Add file-based orchestrator persistence test for agents/discoveries (agents.json + discoveries.json)
- [x] ATOM-008: Add explicit error messages for missing coordination_dir files in FileOrchestrator
- [x] ATOM-009: Add unit tests for AgentConfig defaults and heartbeat thread teardown
- [x] ATOM-010: Add smoke test ensuring async orchestrator remains the default for CLI entrypoints

## P1 - Reliability / Observability / Performance
- [x] ATOM-101: Add ralph test gate summary output to a log file under .coordination/logs
- [x] ATOM-102: Add coverage reports for Option C pytest and Option B jest (store in artifacts/)
- [x] ATOM-103: Add lint/typecheck commands to ralph_config for Option B and key extensions
- [x] ATOM-104: Add task queue size and throughput metrics export for Option C monitoring
- [x] ATOM-105: Add retry/backoff policy unit tests for orchestrator task claiming
- [x] ATOM-106: Add integration test for dependency enforcement across task graphs
- [x] ATOM-107: Add validation to reject duplicate task IDs in FileOrchestrator
- [x] ATOM-108: Add schema validation for tasks.json load (detect malformed entries)
- [x] ATOM-109: Add structured logging format for orchestrator events (JSON lines)
- [x] ATOM-110: Add performance benchmark for task claim/complete cycles
- [x] ATOM-111: Add agent capability matching tests for orchestrator routing
- [x] ATOM-112: Add CLI flag to export ralph_config test gate results as JSON

## P2 - DX / Cleanup / Nice-to-have
- [x] ATOM-201: Add .gitignore entry for .codex/ and vercel-agent-skills/ artifacts
- [x] ATOM-202: Document hybrid orchestrator behavior in README
- [x] ATOM-203: Add quickstart section for ralph_config.json and scripts/ralph_loop.py
- [x] ATOM-204: Add sample CI workflow running ralph test gate
- [x] ATOM-205: Add lint rules for test_doc_generator warnings (dataclass collection)
- [x] ATOM-206: Add optional pre-commit hook for formatting in Option B and Option C
- [x] ATOM-207: Add docs for property-based testing requirements
- [x] ATOM-208: Add dev script to regenerate test catalog markdown
- [x] ATOM-209: Add script to clean coordination artifacts (tasks/logs/results)
- [x] ATOM-210: Add minimal changelog template for v2.1 iterations
