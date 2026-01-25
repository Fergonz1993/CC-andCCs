# TODO_RALPH.md

End goal: Production-ready multi-agent coordination system with real API endpoints, observability stack, containerization, and comprehensive documentation.
Status: v2.2.0 in progress.

---

## v2.2.0 Backlog

### P0 - Production Hardening
- [x] PROD-001: Create Dockerfile for Option C orchestrator
- [x] PROD-002: Create Dockerfile for Option B MCP server
- [x] PROD-003: Add docker-compose.yml for full stack deployment
- [x] PROD-004: Implement rate limiting middleware for MCP server
- [x] PROD-005: Add JWT authentication to Option B endpoints
- [x] PROD-006: Create k6 load testing scripts for task throughput
- [x] PROD-007: Add health check endpoints to all options
- [x] PROD-008: Implement graceful shutdown handlers

### P1 - Observability Stack
- [x] OBS-001: Add Prometheus metrics exporter for Option C
- [x] OBS-002: Create Grafana dashboard JSON for coordination metrics
- [x] OBS-003: Integrate OpenTelemetry tracing for task lifecycle
- [x] OBS-004: Add structured logging with correlation IDs
- [x] OBS-005: Create alerting rules for task queue depth

### P2 - Documentation & Adoption
- [x] DOC-001: Create GitHub Release for v2.1.0 with full release notes
- [x] DOC-002: Set up GitHub Pages documentation site
- [ ] DOC-003: Add interactive API playground (Swagger UI)
- [x] DOC-004: Create video tutorial script and storyboard
- [x] DOC-005: Add architecture diagrams (Mermaid)

### P3 - Cleanup & Maintenance
- [x] CLEAN-001: Archive v2.1.0 completed items to CHANGELOG
- [x] CLEAN-002: Remove stale feature_list.json references
- [x] CLEAN-003: Consolidate duplicate code across options
- [x] CLEAN-004: Update all README files with v2.1.0 features
- [x] CLEAN-005: Add CODEOWNERS file for review assignments

---

## Archived Releases

<details>
<summary><strong>v2.1.0 Completed (32 items)</strong></summary>

### P0 - Correctness / Security / Critical Path (10 items)
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

### P1 - Reliability / Observability / Performance (12 items)
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

### P2 - DX / Cleanup / Nice-to-have (10 items)
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

</details>
