# Workflow Examples

This document presents common workflow patterns using the Claude Multi-Agent Coordination System.

## Table of Contents

1. [Web Application Development](#web-application-development)
2. [API Development](#api-development)
3. [Code Refactoring](#code-refactoring)
4. [Bug Investigation](#bug-investigation)
5. [Documentation Writing](#documentation-writing)
6. [Testing and QA](#testing-and-qa)

---

## Web Application Development

### Goal
Build a todo list web application with React frontend and Node.js backend.

### Task Breakdown

```
Priority 1 (Foundation):
├── Task 1: Set up project structure and tooling
├── Task 2: Design database schema
└── Task 3: Create component architecture

Priority 2 (Core Features):
├── Task 4: Implement database models (depends: 2)
├── Task 5: Create API endpoints (depends: 4)
├── Task 6: Build UI components (depends: 3)
└── Task 7: Implement state management (depends: 6)

Priority 3 (Integration):
├── Task 8: Connect frontend to API (depends: 5, 7)
├── Task 9: Add authentication (depends: 5)
└── Task 10: Implement real-time updates (depends: 8)

Priority 4 (Polish):
├── Task 11: Write unit tests (depends: 5, 6)
├── Task 12: Add error handling (depends: 8)
├── Task 13: Implement responsive design (depends: 6)
└── Task 14: Write documentation (depends: all)
```

### Implementation (Option A)

```bash
# Terminal 1 (Leader)
python coordination.py leader init "Build todo list web app" \
  --approach "React + Node.js + PostgreSQL stack"

# Foundation tasks (no dependencies)
python coordination.py leader add-task "Set up project structure with Vite, Express, and TypeScript" -p 1
python coordination.py leader add-task "Design PostgreSQL schema for todos and users" -p 1
python coordination.py leader add-task "Create React component architecture document" -p 1

# Core features
python coordination.py leader add-task "Implement Prisma models for Todo and User" -p 2 -d task-1 task-2
python coordination.py leader add-task "Create REST API endpoints for CRUD operations" -p 2 -d task-4
python coordination.py leader add-task "Build TodoList, TodoItem, and AddTodo components" -p 2 -d task-3
python coordination.py leader add-task "Set up React Query for state management" -p 2 -d task-6

# Integration
python coordination.py leader add-task "Connect React components to API endpoints" -p 3 -d task-5 task-7
python coordination.py leader add-task "Add JWT authentication flow" -p 3 -d task-5
python coordination.py leader add-task "Implement WebSocket for real-time updates" -p 3 -d task-8

# Polish
python coordination.py leader add-task "Write Jest tests for API endpoints" -p 4 -d task-5
python coordination.py leader add-task "Add comprehensive error handling and toasts" -p 4 -d task-8
python coordination.py leader add-task "Implement responsive Tailwind CSS styling" -p 4 -d task-6
python coordination.py leader add-task "Write API documentation with examples" -p 4
```

### Implementation (Option C)

```python
import asyncio
from orchestrator import Orchestrator

async def build_todo_app():
    orch = Orchestrator(working_directory="./todo-app", max_workers=3)
    await orch.initialize("Build todo list web application")

    # Foundation
    t1 = orch.add_task("Set up project structure", priority=1)
    t2 = orch.add_task("Design database schema", priority=1)
    t3 = orch.add_task("Create component architecture", priority=1)

    # Core
    t4 = orch.add_task("Implement database models", priority=2, dependencies=[t2.id])
    t5 = orch.add_task("Create API endpoints", priority=2, dependencies=[t4.id])
    t6 = orch.add_task("Build UI components", priority=2, dependencies=[t3.id])
    t7 = orch.add_task("Set up state management", priority=2, dependencies=[t6.id])

    # Integration
    t8 = orch.add_task("Connect frontend to API", priority=3, dependencies=[t5.id, t7.id])
    t9 = orch.add_task("Add authentication", priority=3, dependencies=[t5.id])

    # Polish
    orch.add_task("Write tests", priority=4, dependencies=[t5.id, t6.id])
    orch.add_task("Add error handling", priority=4, dependencies=[t8.id])

    result = await orch.run_with_predefined_tasks()
    return result

asyncio.run(build_todo_app())
```

---

## API Development

### Goal
Create a RESTful API for user management with authentication.

### Task Breakdown

```
Phase 1: Design
├── Task 1: Define API endpoints and methods
├── Task 2: Design request/response schemas
└── Task 3: Plan authentication strategy

Phase 2: Implementation
├── Task 4: Set up Express/Fastify server (depends: 1)
├── Task 5: Implement user CRUD endpoints (depends: 2, 4)
├── Task 6: Add JWT authentication (depends: 3, 4)
├── Task 7: Implement input validation (depends: 5)
└── Task 8: Add rate limiting and security (depends: 6)

Phase 3: Quality
├── Task 9: Write API tests (depends: 5, 6)
├── Task 10: Generate OpenAPI documentation (depends: 5)
└── Task 11: Set up CI/CD pipeline (depends: 9)
```

### Example Output

After completion, the coordination produces:

**results/task-5.md:**
```markdown
# Task: Implement user CRUD endpoints

## Description
Create REST endpoints for user management

## Approach
Implemented using Express.js with TypeScript:
- GET /users - List all users (paginated)
- GET /users/:id - Get user by ID
- POST /users - Create new user
- PUT /users/:id - Update user
- DELETE /users/:id - Delete user

## Files Created
- src/routes/users.ts
- src/controllers/userController.ts
- src/middleware/validate.ts

## Files Modified
- src/routes/index.ts (added user routes)

## Status
SUCCESS

## Notes
Used Zod for request validation. Pagination defaults to 20 items.
```

---

## Code Refactoring

### Goal
Refactor legacy authentication module to modern patterns.

### Task Breakdown

```
Analysis Phase:
├── Task 1: Document current auth implementation
├── Task 2: Identify code smells and technical debt
└── Task 3: Design new architecture

Refactoring Phase:
├── Task 4: Create new auth module structure (depends: 3)
├── Task 5: Migrate login functionality (depends: 4)
├── Task 6: Migrate logout functionality (depends: 4)
├── Task 7: Migrate session management (depends: 5, 6)
├── Task 8: Migrate password reset (depends: 7)
└── Task 9: Update all call sites (depends: 5, 6, 7, 8)

Validation Phase:
├── Task 10: Write comprehensive tests (depends: 9)
├── Task 11: Performance comparison (depends: 9)
└── Task 12: Remove legacy code (depends: 10, 11)
```

### Discovery Sharing Example

During refactoring, workers share discoveries:

```bash
# Worker finds existing utility
# In Option A:
echo "Found reusable hash function in src/utils/crypto.ts" >> .coordination/context/discoveries.md

# In Option B:
# Use add_discovery with content="Found reusable hash function..." tags=["auth", "utils"]
```

---

## Bug Investigation

### Goal
Investigate and fix slow API response times.

### Task Breakdown

```
Investigation:
├── Task 1: Profile endpoint performance
├── Task 2: Analyze database query execution plans
├── Task 3: Review application logs for patterns
└── Task 4: Check for N+1 query issues

Root Cause Analysis:
├── Task 5: Compile findings report (depends: 1, 2, 3, 4)
└── Task 6: Prioritize fixes by impact (depends: 5)

Fixes:
├── Task 7: Implement identified optimizations (depends: 6)
├── Task 8: Add database indexes (depends: 6)
├── Task 9: Implement query caching (depends: 7)
└── Task 10: Add connection pooling (depends: 7)

Validation:
├── Task 11: Load test improvements (depends: 7, 8, 9, 10)
└── Task 12: Document changes and metrics (depends: 11)
```

### Worker Context Hints

```bash
python coordination.py leader add-task "Profile endpoint performance" -p 1 \
  -f src/routes/api.ts src/middleware/timing.ts \
  --hints "Use clinic.js or 0x for profiling. Focus on /api/search endpoint."

python coordination.py leader add-task "Analyze database queries" -p 1 \
  -f prisma/schema.prisma \
  --hints "Run EXPLAIN ANALYZE on slow queries. Check for missing indexes."
```

---

## Documentation Writing

### Goal
Create comprehensive documentation for an open-source library.

### Task Breakdown

```
Foundation:
├── Task 1: Set up documentation site (Docusaurus/VitePress)
├── Task 2: Create documentation structure outline
└── Task 3: Write style guide for consistency

Core Documentation:
├── Task 4: Write getting started guide (depends: 1)
├── Task 5: Document API reference (depends: 2)
├── Task 6: Create tutorial series (depends: 4)
├── Task 7: Write configuration guide (depends: 5)
└── Task 8: Add troubleshooting section (depends: 4, 5)

Examples & Extras:
├── Task 9: Create code examples (depends: 5)
├── Task 10: Write migration guide (depends: 7)
├── Task 11: Add FAQ section (depends: 8)
└── Task 12: Review and polish all docs (depends: all)
```

### Parallel Documentation Work

```
Worker 1: API Reference (technical, needs code analysis)
Worker 2: Tutorials (user-focused, needs testing)
Worker 3: Examples (code-focused, needs validation)
```

---

## Testing and QA

### Goal
Implement comprehensive test suite for an application.

### Task Breakdown

```
Setup:
├── Task 1: Configure Jest/Vitest testing framework
├── Task 2: Set up test database and fixtures
└── Task 3: Create test utilities and helpers

Unit Tests:
├── Task 4: Write unit tests for models (depends: 1, 2)
├── Task 5: Write unit tests for services (depends: 1, 3)
├── Task 6: Write unit tests for utilities (depends: 1)

Integration Tests:
├── Task 7: Write API integration tests (depends: 2, 4, 5)
├── Task 8: Write database integration tests (depends: 2, 4)

E2E Tests:
├── Task 9: Set up Playwright/Cypress (depends: 1)
├── Task 10: Write E2E tests for critical paths (depends: 9)

Quality:
├── Task 11: Achieve 80% code coverage (depends: 4, 5, 6, 7, 8)
├── Task 12: Set up CI test pipeline (depends: 11)
└── Task 13: Write test documentation (depends: all)
```

### Test Distribution Strategy

- **Worker 1**: Unit tests (fast, independent)
- **Worker 2**: Integration tests (need database)
- **Worker 3**: E2E tests (need full environment)

This parallelization maximizes throughput by separating concerns and resource requirements.
