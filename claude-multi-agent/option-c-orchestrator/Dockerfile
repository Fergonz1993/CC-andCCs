# Claude Multi-Agent Orchestrator - Option C
# Production Dockerfile for Python orchestrator service
#
# Build: docker build -t claude-orchestrator .
# Run:   docker run -v $(pwd)/.coordination:/app/.coordination claude-orchestrator

# =============================================================================
# Stage 1: Build dependencies
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only dependency files first (for layer caching)
COPY pyproject.toml ./

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir build hatchling && \
    pip install --no-cache-dir .

# =============================================================================
# Stage 2: Production image
# =============================================================================
FROM python:3.12-slim AS production

# Security: Run as non-root user
RUN groupadd --gid 1000 orchestrator && \
    useradd --uid 1000 --gid orchestrator --shell /bin/bash --create-home orchestrator

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY --chown=orchestrator:orchestrator src/ ./src/
COPY --chown=orchestrator:orchestrator pyproject.toml ./

# Install the package in editable mode for proper module resolution
RUN pip install --no-cache-dir -e .

# Create coordination directory with proper permissions
RUN mkdir -p /app/.coordination && chown orchestrator:orchestrator /app/.coordination

# Environment variables with defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    COORDINATION_DIR=/app/.coordination \
    ORCHESTRATOR_MAX_WORKERS=3 \
    ORCHESTRATOR_TASK_TIMEOUT=600 \
    ORCHESTRATOR_MAX_ATTEMPTS=3 \
    CLAUDE_MODEL=claude-sonnet-4-20250514

# Health check port (for future HTTP health endpoint)
ENV HEALTH_PORT=8080
EXPOSE 8080

# Volume for coordination state
VOLUME ["/app/.coordination"]

# Switch to non-root user
USER orchestrator

# Health check - verify the CLI is functional
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from orchestrator import Orchestrator; print('healthy')" || exit 1

# Default command - show help (override with actual command)
ENTRYPOINT ["orchestrate"]
CMD ["--help"]

# =============================================================================
# Labels for container metadata
# =============================================================================
LABEL org.opencontainers.image.title="Claude Multi-Agent Orchestrator" \
      org.opencontainers.image.description="External process orchestrator for multi-agent Claude Code coordination" \
      org.opencontainers.image.version="2.1.0" \
      org.opencontainers.image.vendor="Claude Code" \
      org.opencontainers.image.source="https://github.com/anthropics/claude-code" \
      org.opencontainers.image.licenses="MIT"
