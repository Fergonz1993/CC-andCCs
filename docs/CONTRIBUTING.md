# Contributing Guidelines

Thank you for your interest in contributing to the Claude Multi-Agent Coordination System! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How to Contribute](#how-to-contribute)
4. [Development Process](#development-process)
5. [Code Style](#code-style)
6. [Testing](#testing)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)
9. [Release Process](#release-process)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive experience for everyone. We expect all contributors to:

- Be respectful and considerate
- Give and receive constructive feedback gracefully
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Public or private harassment
- Publishing others' private information without permission

---

## Getting Started

### Prerequisites

- Git
- Python 3.8+ (for Options A and C)
- Node.js 18+ (for Option B)
- Claude Code (for testing)

### Setting Up Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd cc-and-ccs

# Set up Option A
cd claude-multi-agent/option-a-file-based
# No additional setup needed

# Set up Option B
cd claude-multi-agent/option-b-mcp-broker
npm install
npm run build

# Set up Option C
cd claude-multi-agent/option-c-orchestrator
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Running Tests

```bash
# Option A (no tests yet - contributions welcome!)

# Option B
cd option-b-mcp-broker
npm test

# Option C
cd option-c-orchestrator
pytest
```

---

## How to Contribute

### Reporting Bugs

Before submitting a bug report:

1. Check existing issues for duplicates
2. Collect diagnostic information

Bug reports should include:

```markdown
## Description
[Clear description of the bug]

## Steps to Reproduce
1. [First Step]
2. [Second Step]
3. [...]

## Expected Behavior
[What you expected to happen]

## Actual Behavior
[What actually happened]

## Environment
- OS: [e.g., macOS 13.0]
- Python Version: [e.g., 3.11.0]
- Node.js Version: [e.g., 18.12.0]
- Option: [A/B/C]

## Additional Context
[Logs, screenshots, etc.]
```

### Suggesting Features

Feature requests should include:

```markdown
## Problem Statement
[What problem does this solve?]

## Proposed Solution
[How would it work?]

## Alternatives Considered
[Other approaches you've thought about]

## Use Cases
[Who would benefit and how?]
```

### Contributing Code

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

### Contributing Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples
- Improve structure
- Translate to other languages

---

## Development Process

### Branching Strategy

```
main
├── develop
│   ├── feature/new-feature
│   ├── feature/another-feature
│   └── bugfix/fix-something
└── release/v1.1.0
```

- `main`: Stable release branch
- `develop`: Integration branch for features
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `release/*`: Release preparation

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(option-a): add task tagging support

Add ability to tag tasks for filtering and organization.
Includes CLI support and schema update.

Closes #123
```

```
fix(option-b): prevent race condition in claim_task

Re-read state after claiming to verify success.
Adds retry logic if claim fails.
```

---

## Code Style

### Python (Options A & C)

Follow [PEP 8](https://pep8.org/) with these additions:

```python
# Use type hints
def add_task(description: str, priority: int = 5) -> Task:
    ...

# Use docstrings
def claim_task(agent_id: str) -> Optional[Task]:
    """
    Claim an available task for the specified agent.

    Args:
        agent_id: Unique identifier for the claiming agent

    Returns:
        The claimed Task, or None if no tasks available

    Raises:
        CoordinationError: If coordination not initialized
    """
    ...

# Use constants for magic values
PRIORITY_BOOST_THRESHOLD_MINUTES = 5
DEFAULT_TASK_TIMEOUT_SECONDS = 600
```

Tools:
- `black` for formatting
- `isort` for import sorting
- `mypy` for type checking
- `pylint` for linting

```bash
# Format code
black .
isort .

# Check types
mypy src/

# Lint
pylint src/
```

### TypeScript (Option B)

Follow the existing style with ESLint and Prettier:

```typescript
// Use TypeScript types
interface Task {
  id: string;
  description: string;
  status: TaskStatus;
  priority: number;
}

// Use const assertions where appropriate
const TASK_STATUSES = ['available', 'claimed', 'in_progress', 'done', 'failed'] as const;
type TaskStatus = typeof TASK_STATUSES[number];

// Document functions
/**
 * Claim an available task for the specified agent.
 * @param agentId - Unique identifier for the claiming agent
 * @returns The claimed task, or null if none available
 */
function claimTask(agentId: string): Task | null {
  ...
}
```

Tools:
- `prettier` for formatting
- `eslint` for linting
- `typescript` for type checking

```bash
npm run lint
npm run format
npm run typecheck
```

---

## Testing

### Test Requirements

- All new features should include tests
- Bug fixes should include regression tests
- Maintain or improve code coverage

### Python Testing

```python
# tests/test_coordination.py
import pytest
from coordination import Task, leader_add_task, worker_claim

@pytest.fixture
def coordination_setup(tmp_path):
    """Set up a temporary coordination directory."""
    coord_dir = tmp_path / ".coordination"
    coord_dir.mkdir()
    # ... setup ...
    yield coord_dir
    # ... cleanup ...

def test_add_task(coordination_setup):
    """Test adding a task to the queue."""
    task_id = leader_add_task("Test task", priority=1)
    assert task_id.startswith("task-")

def test_claim_with_dependencies(coordination_setup):
    """Test that tasks with unmet dependencies cannot be claimed."""
    task1 = leader_add_task("Task 1", priority=1)
    task2 = leader_add_task("Task 2", priority=2, dependencies=[task1])

    claimed = worker_claim("worker-1")
    assert claimed.id == task1  # Task 2 blocked by dependency
```

### TypeScript Testing

```typescript
// src/__tests__/coordination.test.ts
import { createTask, claimTask, getAvailableTasks } from '../index';

describe('Task Management', () => {
  beforeEach(() => {
    // Reset state
  });

  test('createTask generates unique IDs', () => {
    const task1 = createTask('Task 1');
    const task2 = createTask('Task 2');
    expect(task1.id).not.toBe(task2.id);
  });

  test('claimTask respects dependencies', () => {
    const task1 = createTask('Task 1');
    const task2 = createTask('Task 2', 5, [task1.id]);

    const claimed = claimTask('worker-1');
    expect(claimed?.id).toBe(task1.id);
  });
});
```

---

## Documentation

### Documentation Requirements

- All public APIs must be documented
- Include examples for non-trivial features
- Keep README files up to date
- Update changelog for notable changes

### Documentation Style

```markdown
# Feature Name

Brief description of what this feature does.

## Usage

```python
# Example code
result = do_something(param)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| param | str | required | Description of param |

## Returns

Description of return value.

## Examples

### Basic Example

```python
# Simple usage
```

### Advanced Example

```python
# Complex usage with options
```

## See Also

- [Related Feature](link)
- [API Reference](link)
```

---

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] Changelog updated (for notable changes)
- [ ] Commits are clean and descriptive

### PR Template

```markdown
## Description
[What does this PR do?]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
[How was this tested?]

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Changelog updated

## Related Issues
Closes #[issue number]
```

### Review Process

1. Automated checks must pass
2. At least one maintainer review required
3. Address all review comments
4. Squash commits if requested
5. Maintainer merges when approved

---

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. [ ] All tests passing
2. [ ] Changelog updated
3. [ ] Version bumped in relevant files
4. [ ] Documentation reviewed
5. [ ] Release notes drafted
6. [ ] Tag created
7. [ ] Packages published

---

## Recognition

Contributors will be recognized in:

- GitHub contributors list
- Release notes
- Documentation acknowledgments

Thank you for contributing!
