#!/bin/bash
#
# File-Based Coordination System Demo (Option A)
# ==============================================
#
# This script demonstrates the file-based multi-agent coordination system.
# It shows how multiple Claude Code instances can coordinate work through
# a shared JSON file without requiring any external infrastructure.
#
# The coordination system provides:
# - Session management with goals and context
# - Task creation with dependencies and priorities
# - Agent registration and heartbeat tracking
# - Status reporting across all agents
#
# Usage: ./demo.sh
#

set -e

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COORD_SCRIPT="$SCRIPT_DIR/coordination.py"

echo "========================================"
echo "File-Based Coordination System Demo"
echo "========================================"
echo ""

# Step 1: Initialize a new coordination session
echo "Step 1: Initializing coordination session..."
echo "--------------------------------------------"
echo "This creates a new session with a shared goal that all agents work toward."
echo ""

python3 "$COORD_SCRIPT" leader init \
    "Build a REST API with authentication and database integration" \
    --approach "Tech stack: Python/FastAPI, PostgreSQL, JWT auth. Priority: Security first."

echo ""
echo "Session initialized! A coordination.json file has been created."
echo ""

# Step 2: Add tasks with dependencies
echo "Step 2: Adding tasks with dependencies..."
echo "-----------------------------------------"
echo "Tasks can depend on other tasks. Agents will only pick up tasks"
echo "whose dependencies are complete."
echo ""

# Task 1: Database schema (no dependencies - can start immediately)
echo "Adding Task 1: Database Schema Design (no dependencies)..."
python3 "$COORD_SCRIPT" leader add-task \
    "Design and implement database schema for users, sessions, and resources" \
    -p 1

echo ""

# Task 2: Auth module (depends on database)
echo "Adding Task 2: Authentication Module (depends on Task 1)..."
python3 "$COORD_SCRIPT" leader add-task \
    "Implement JWT-based authentication with login, logout, and token refresh" \
    -p 1

echo ""

# Task 3: API endpoints (depends on auth)
echo "Adding Task 3: API Endpoints (depends on Task 2)..."
python3 "$COORD_SCRIPT" leader add-task \
    "Create CRUD endpoints for resources with proper auth middleware" \
    -p 2

echo ""

# Task 4: Testing (depends on API endpoints)
echo "Adding Task 4: Integration Tests (depends on Task 3)..."
python3 "$COORD_SCRIPT" leader add-task \
    "Write integration tests covering auth flow and all API endpoints" \
    -p 2

echo ""

# Step 3: Register an agent
echo "Step 3: Registering an agent..."
echo "--------------------------------"
echo "Each Claude Code instance registers as an agent with its capabilities."
echo ""

python3 "$COORD_SCRIPT" register \
    --agent-id "demo-agent-1" \
    --capabilities "python,fastapi,postgresql,testing"

echo ""

# Step 4: Show the current status
echo "Step 4: Displaying coordination status..."
echo "-----------------------------------------"
echo ""

python3 "$COORD_SCRIPT" leader status

echo ""
echo "========================================"
echo "Demo Complete!"
echo "========================================"
echo ""
echo "What you can do next:"
echo ""
echo "  1. Claim a task:     python3 $COORD_SCRIPT worker claim demo-agent-1"
echo "  2. Complete a task:  python3 $COORD_SCRIPT worker complete demo-agent-1 task-xxx 'Task output summary'"
echo "  3. Add more tasks:   python3 $COORD_SCRIPT leader add-task 'Task description' -p 3"
echo "  4. Check status:     python3 $COORD_SCRIPT leader status"
echo "  5. List tasks:       python3 $COORD_SCRIPT worker list"
echo ""
echo "The coordination.json file in this directory contains all session data."
echo "Multiple Claude Code instances can read/write to coordinate their work."
echo ""
