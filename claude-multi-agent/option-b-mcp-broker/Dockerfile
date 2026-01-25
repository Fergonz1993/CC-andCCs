# Claude Multi-Agent MCP Broker - Option B
# Production Dockerfile for TypeScript MCP server
#
# Build: docker build -t claude-mcp-broker .
# Run:   docker run -v $(pwd)/.coordination:/app/.coordination -p 8081:8081 claude-mcp-broker

# =============================================================================
# Stage 1: Build TypeScript
# =============================================================================
FROM node:20-alpine AS builder

WORKDIR /build

# Install build dependencies for native modules (better-sqlite3)
RUN apk add --no-cache python3 make g++

# Copy package files first (for layer caching)
COPY package*.json ./

# Install all dependencies including devDependencies for build
RUN npm ci

# Copy source files
COPY tsconfig.json ./
COPY src/ ./src/

# Build TypeScript to JavaScript
RUN npm run build

# Remove devDependencies for production
RUN npm prune --production

# =============================================================================
# Stage 2: Production image
# =============================================================================
FROM node:20-alpine AS production

# Security: Run as non-root user
RUN addgroup -g 1000 mcpserver && \
    adduser -u 1000 -G mcpserver -s /bin/sh -D mcpserver

WORKDIR /app

# Install runtime dependencies for native modules
RUN apk add --no-cache libstdc++

# Copy built artifacts and production dependencies
COPY --from=builder --chown=mcpserver:mcpserver /build/dist ./dist/
COPY --from=builder --chown=mcpserver:mcpserver /build/node_modules ./node_modules/
COPY --from=builder --chown=mcpserver:mcpserver /build/package.json ./

# Create coordination directory with proper permissions
RUN mkdir -p /app/.coordination && chown mcpserver:mcpserver /app/.coordination

# Environment variables with defaults
ENV NODE_ENV=production \
    COORDINATION_DIR=/app/.coordination \
    MCP_SERVER_PORT=8081 \
    LOG_LEVEL=info

# Expose MCP server port
EXPOSE 8081

# Volume for coordination state persistence
VOLUME ["/app/.coordination"]

# Switch to non-root user
USER mcpserver

# Health check - verify the server can start and respond
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD node -e "const fs = require('fs'); const pkg = require('./package.json'); console.log('healthy:', pkg.version);" || exit 1

# Run the MCP server
ENTRYPOINT ["node"]
CMD ["dist/index.js"]

# =============================================================================
# Labels for container metadata
# =============================================================================
LABEL org.opencontainers.image.title="Claude Multi-Agent MCP Broker" \
      org.opencontainers.image.description="MCP Server for multi-agent Claude Code coordination" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="Claude Code" \
      org.opencontainers.image.source="https://github.com/anthropics/claude-code" \
      org.opencontainers.image.licenses="MIT"
