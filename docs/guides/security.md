# Security Best Practices Guide

This guide covers security considerations and best practices for deploying and operating the Claude Multi-Agent Coordination System.

## Table of Contents

1. [Security Model Overview](#security-model-overview)
2. [Authentication & Authorization](#authentication--authorization)
3. [Data Protection](#data-protection)
4. [Network Security](#network-security)
5. [Input Validation](#input-validation)
6. [Audit & Logging](#audit--logging)
7. [Deployment Security](#deployment-security)
8. [Security Checklist](#security-checklist)

---

## Security Model Overview

### Trust Boundaries

```
┌─────────────────────────────────────────────────────────────┐
│                    Trusted Zone                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Leader     │  │  Worker 1   │  │  Worker 2   │        │
│  │  Agent      │  │  Agent      │  │  Agent      │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │
│         └────────────────┼────────────────┘                │
│                          │                                  │
│                 ┌────────▼────────┐                        │
│                 │  Coordination   │                        │
│                 │     Layer       │                        │
│                 └────────┬────────┘                        │
│                          │                                  │
├──────────────────────────┼──────────────────────────────────┤
│                    ┌─────▼─────┐                           │
│                    │ Filesystem│  Potentially Shared       │
│                    └───────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### Threat Model

| Threat | Impact | Mitigation |
|--------|--------|------------|
| Unauthorized task creation | Medium | Agent authentication |
| Task data tampering | High | File permissions, integrity checks |
| Credential exposure | Critical | Secrets management |
| Denial of service | Medium | Rate limiting, resource limits |
| Information disclosure | Medium | Access control, encryption |
| Malicious task execution | High | Input validation, sandboxing |

---

## Authentication & Authorization

### Option A: File-Based

File-based coordination relies on filesystem permissions for access control.

**Best Practices:**

```bash
# Restrict coordination directory access
chmod 700 .coordination
chown $(whoami) .coordination

# Make tasks file readable/writable only by owner
chmod 600 .coordination/tasks.json

# If sharing between users, use group permissions
chmod 770 .coordination
chgrp dev-team .coordination
```

**Agent Identification:**

```python
# Use unique, verifiable agent IDs
import hashlib
import os

def generate_agent_id():
    """Generate a unique agent ID based on terminal and user."""
    user = os.getenv('USER', 'unknown')
    pid = os.getpid()
    term = os.getenv('TERM_SESSION_ID', os.getenv('WINDOWID', str(pid)))
    unique = f"{user}-{term}-{pid}"
    return f"agent-{hashlib.sha256(unique.encode()).hexdigest()[:8]}"
```

### Option B: MCP Server

**API Key Authentication:**

```typescript
// Add to mcp.json environment
{
  "env": {
    "COORDINATION_API_KEY": "your-secret-key-here"
  }
}

// Validate in server
function validateApiKey(request: any): boolean {
  const key = request.headers?.['x-api-key'];
  return key === process.env.COORDINATION_API_KEY;
}
```

**Role-Based Access Control:**

```typescript
const PERMISSIONS = {
  leader: ['init_coordination', 'create_task', 'create_tasks_batch', 'get_results'],
  worker: ['claim_task', 'start_task', 'complete_task', 'fail_task', 'heartbeat'],
  all: ['get_status', 'get_all_tasks', 'add_discovery', 'get_discoveries', 'get_master_plan']
};

function checkPermission(agentId: string, tool: string): boolean {
  const agent = state.agents.get(agentId);
  if (!agent) return false;

  const allowed = [...PERMISSIONS[agent.role], ...PERMISSIONS.all];
  return allowed.includes(tool);
}
```

### Option C: Orchestrator

**Secure Configuration:**

```python
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class SecurityConfig:
    # Require authentication
    require_auth: bool = True

    # API key for external access
    api_key: Optional[str] = None

    # Allowed agent IDs (None = allow all)
    allowed_agents: Optional[list[str]] = None

    # Maximum tasks per session
    max_tasks: int = 1000

    # Task description max length
    max_description_length: int = 10000

    @classmethod
    def from_env(cls):
        return cls(
            require_auth=os.getenv('REQUIRE_AUTH', 'true').lower() == 'true',
            api_key=os.getenv('COORDINATION_API_KEY'),
            allowed_agents=os.getenv('ALLOWED_AGENTS', '').split(',') or None,
            max_tasks=int(os.getenv('MAX_TASKS', '1000')),
        )
```

---

## Data Protection

### Sensitive Data Handling

**Never store in tasks:**
- API keys or tokens
- Passwords or credentials
- Personal identifying information (PII)
- Encryption keys

**Use environment variables instead:**

```bash
# Good: Reference secrets by name
python coordination.py leader add-task "Configure API using DB_PASSWORD env var"

# Bad: Include secrets in task
python coordination.py leader add-task "Configure API with password=secret123"
```

### Encryption at Rest

**For sensitive coordination data:**

```python
from cryptography.fernet import Fernet
import json
import os

class EncryptedStorage:
    def __init__(self, key_path: str = '.coordination/key'):
        if os.path.exists(key_path):
            with open(key_path, 'rb') as f:
                self.key = f.read()
        else:
            self.key = Fernet.generate_key()
            with open(key_path, 'wb') as f:
                f.write(self.key)
            os.chmod(key_path, 0o600)

        self.cipher = Fernet(self.key)

    def save(self, filepath: str, data: dict):
        encrypted = self.cipher.encrypt(json.dumps(data).encode())
        with open(filepath, 'wb') as f:
            f.write(encrypted)

    def load(self, filepath: str) -> dict:
        with open(filepath, 'rb') as f:
            encrypted = f.read()
        decrypted = self.cipher.decrypt(encrypted)
        return json.loads(decrypted.decode())
```

### Data Retention

```python
from datetime import datetime, timedelta

def cleanup_old_data(max_age_days: int = 30):
    """Remove old task data beyond retention period."""
    cutoff = datetime.now() - timedelta(days=max_age_days)

    data = load_tasks()

    # Archive old completed tasks
    archived = []
    remaining = []

    for task in data['tasks']:
        completed_at = task.get('completed_at')
        if completed_at and datetime.fromisoformat(completed_at) < cutoff:
            archived.append(task)
        else:
            remaining.append(task)

    # Save archive
    if archived:
        archive_file = f".coordination/archive-{datetime.now().strftime('%Y%m%d')}.json"
        with open(archive_file, 'w') as f:
            json.dump({'tasks': archived}, f)

    # Update active tasks
    data['tasks'] = remaining
    save_tasks(data)

    return len(archived)
```

---

## Network Security

### Option B: MCP Server

**Use TLS for HTTP Transport:**

```typescript
import https from 'https';
import fs from 'fs';

const server = https.createServer({
  key: fs.readFileSync('server.key'),
  cert: fs.readFileSync('server.crt'),
}, app);
```

**IP Allowlisting:**

```typescript
const ALLOWED_IPS = process.env.ALLOWED_IPS?.split(',') || ['127.0.0.1', '::1'];

function ipAllowlistMiddleware(req: any, res: any, next: any) {
  const clientIp = req.ip || req.connection.remoteAddress;
  if (!ALLOWED_IPS.includes(clientIp)) {
    return res.status(403).json({ error: 'IP not allowed' });
  }
  next();
}
```

### Rate Limiting

```typescript
const rateLimit = new Map<string, { count: number; resetAt: number }>();

function rateLimitMiddleware(req: any, res: any, next: any) {
  const agentId = req.headers['x-agent-id'] || 'anonymous';
  const now = Date.now();
  const windowMs = 60 * 1000; // 1 minute
  const maxRequests = 100;

  let record = rateLimit.get(agentId);
  if (!record || record.resetAt < now) {
    record = { count: 0, resetAt: now + windowMs };
    rateLimit.set(agentId, record);
  }

  record.count++;
  if (record.count > maxRequests) {
    return res.status(429).json({
      error: 'Rate limit exceeded',
      retryAfter: Math.ceil((record.resetAt - now) / 1000)
    });
  }

  next();
}
```

---

## Input Validation

### Task Description Sanitization

```python
import re
import html

def sanitize_task_description(description: str) -> str:
    """Sanitize task description to prevent injection attacks."""
    # Remove control characters
    description = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', description)

    # Escape HTML entities
    description = html.escape(description)

    # Limit length
    max_length = 10000
    if len(description) > max_length:
        description = description[:max_length] + '...'

    return description
```

### Path Validation

```python
import os
from pathlib import Path

def validate_file_path(filepath: str, base_dir: str = '.') -> bool:
    """Ensure file path is within allowed directory."""
    try:
        base = Path(base_dir).resolve()
        target = Path(filepath).resolve()

        # Check if target is under base directory
        target.relative_to(base)
        return True
    except (ValueError, OSError):
        return False

# Usage
context_files = task.get('context', {}).get('files', [])
valid_files = [f for f in context_files if validate_file_path(f, working_dir)]
```

### JSON Schema Validation

```python
import jsonschema

TASK_SCHEMA = {
    "type": "object",
    "required": ["description", "priority"],
    "properties": {
        "description": {
            "type": "string",
            "minLength": 1,
            "maxLength": 10000
        },
        "priority": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10
        },
        "dependencies": {
            "type": "array",
            "items": {"type": "string", "pattern": "^task-[a-z0-9-]+$"},
            "maxItems": 100
        }
    },
    "additionalProperties": False
}

def validate_task(task_data: dict) -> bool:
    try:
        jsonschema.validate(task_data, TASK_SCHEMA)
        return True
    except jsonschema.ValidationError as e:
        print(f"Validation error: {e.message}")
        return False
```

---

## Audit & Logging

### Comprehensive Audit Trail

```python
import logging
import json
from datetime import datetime

class AuditLogger:
    def __init__(self, log_file: str = '.coordination/audit.log'):
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def log(self, event_type: str, agent_id: str, details: dict):
        record = {
            'timestamp': datetime.now().isoformat(),
            'event': event_type,
            'agent': agent_id,
            **details
        }
        self.logger.info(json.dumps(record))

# Usage
audit = AuditLogger()

def claim_task(agent_id: str, task_id: str):
    # ... claim logic ...
    audit.log('TASK_CLAIMED', agent_id, {'task_id': task_id})

def complete_task(agent_id: str, task_id: str, result: dict):
    # ... complete logic ...
    audit.log('TASK_COMPLETED', agent_id, {
        'task_id': task_id,
        'files_modified': result.get('files_modified', [])
    })
```

### Security Event Monitoring

```python
SECURITY_EVENTS = [
    'AUTH_FAILURE',
    'PERMISSION_DENIED',
    'RATE_LIMIT_EXCEEDED',
    'INVALID_INPUT',
    'SUSPICIOUS_ACTIVITY'
]

def log_security_event(event: str, agent_id: str, details: dict):
    """Log security-relevant events for monitoring."""
    if event in SECURITY_EVENTS:
        audit.log(f'SECURITY_{event}', agent_id, {
            'severity': 'HIGH' if event in ['AUTH_FAILURE', 'SUSPICIOUS_ACTIVITY'] else 'MEDIUM',
            **details
        })
```

---

## Deployment Security

### Production Checklist

```bash
# 1. Set restrictive file permissions
chmod 700 .coordination
chmod 600 .coordination/*.json

# 2. Use dedicated service account
sudo useradd -r -s /bin/false coordination-service
sudo chown -R coordination-service:coordination-service .coordination

# 3. Enable audit logging
export AUDIT_LOG_ENABLED=true

# 4. Set rate limits
export RATE_LIMIT_REQUESTS=100
export RATE_LIMIT_WINDOW_MS=60000

# 5. Configure TLS (Option B)
export TLS_CERT=/path/to/cert.pem
export TLS_KEY=/path/to/key.pem
```

### Container Security (Docker)

```dockerfile
FROM node:18-alpine

# Don't run as root
RUN addgroup -S coordination && adduser -S coordination -G coordination

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy application
COPY --chown=coordination:coordination . .

# Switch to non-root user
USER coordination

# Set secure permissions
RUN chmod 700 /app && chmod 600 /app/*.json

# Health check
HEALTHCHECK CMD curl -f http://localhost:3000/health || exit 1

# Run
CMD ["node", "dist/index.js"]
```

---

## Security Checklist

### Before Deployment

- [ ] File permissions are restrictive (700/600)
- [ ] No secrets in task descriptions or code
- [ ] Input validation enabled for all inputs
- [ ] Rate limiting configured
- [ ] Audit logging enabled
- [ ] TLS enabled for network transport
- [ ] API keys/tokens are securely stored
- [ ] Access control rules defined

### Ongoing Operations

- [ ] Review audit logs regularly
- [ ] Rotate API keys periodically
- [ ] Update dependencies for security patches
- [ ] Monitor for unusual activity
- [ ] Test backup and recovery procedures
- [ ] Review and prune old data

### Incident Response

- [ ] Document security contacts
- [ ] Have incident response plan
- [ ] Know how to revoke access quickly
- [ ] Have backup of clean state
- [ ] Document recovery procedures
