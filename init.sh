#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "=== CC-and-CCs Development Environment Setup ==="
echo "Project Root: $PROJECT_ROOT"
echo ""

# ============================================
# Option A: File-Based (Python)
# ============================================
echo "[1/4] Setting up Option A (File-Based)..."
cd "$PROJECT_ROOT/claude-multi-agent/option-a-file-based"

# Make scripts executable
if [ -f "coordination.py" ]; then
    chmod +x coordination.py
fi
if [ -f "worker-loop.sh" ]; then
    chmod +x worker-loop.sh
fi
if [ -f "demo.sh" ]; then
    chmod +x demo.sh
fi
echo "  ✓ Scripts made executable"

# ============================================
# Option B: MCP Broker (Node.js)
# ============================================
echo "[2/4] Setting up Option B (MCP Broker)..."
cd "$PROJECT_ROOT/claude-multi-agent/option-b-mcp-broker"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "  Installing npm dependencies..."
    npm install --silent
fi

# Build TypeScript
echo "  Building TypeScript..."
npm run build --silent 2>/dev/null || npm run build
echo "  ✓ TypeScript compiled"

# ============================================
# Option C: Orchestrator (Python)
# ============================================
echo "[3/4] Setting up Option C (Orchestrator)..."
cd "$PROJECT_ROOT/claude-multi-agent/option-c-orchestrator"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate and install
echo "  Installing Python dependencies..."
source .venv/bin/activate
pip install --upgrade pip -q
pip install -e ".[dev]" -q 2>/dev/null || pip install -e . -q
deactivate
echo "  ✓ Python package installed"

# ============================================
# Verification
# ============================================
echo "[4/4] Verifying setup..."
echo ""

ERRORS=0

# Check Claude CLI
if command -v claude &> /dev/null; then
    echo "  ✓ Claude CLI found: $(which claude)"
else
    echo "  ⚠️  Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
    ERRORS=$((ERRORS + 1))
fi

# Check Option A
if [ -f "$PROJECT_ROOT/claude-multi-agent/option-a-file-based/coordination.py" ]; then
    echo "  ✓ Option A: coordination.py exists"
else
    echo "  ✗ Option A: coordination.py missing"
    ERRORS=$((ERRORS + 1))
fi

# Check Option B build
if [ -f "$PROJECT_ROOT/claude-multi-agent/option-b-mcp-broker/dist/index.js" ]; then
    echo "  ✓ Option B: dist/index.js built"
else
    echo "  ✗ Option B: build failed (dist/index.js missing)"
    ERRORS=$((ERRORS + 1))
fi

# Check Option C
cd "$PROJECT_ROOT/claude-multi-agent/option-c-orchestrator"
source .venv/bin/activate
if python -c "from orchestrator import Orchestrator; print('OK')" 2>/dev/null; then
    echo "  ✓ Option C: orchestrator package importable"
else
    echo "  ✗ Option C: import failed"
    ERRORS=$((ERRORS + 1))
fi

# Check CLI
if python -c "from orchestrator.cli import app" 2>/dev/null; then
    echo "  ✓ Option C: CLI importable"
fi
deactivate

# ============================================
# Create progress file if missing
# ============================================
cd "$PROJECT_ROOT"
if [ ! -f "claude-progress.txt" ]; then
    echo ""
    echo "  Creating claude-progress.txt..."
    touch claude-progress.txt
fi

# ============================================
# Summary
# ============================================
echo ""
echo "=========================================="

if [ $ERRORS -eq 0 ]; then
    echo "=== Setup Complete - All checks passed ==="
else
    echo "=== Setup Complete - $ERRORS warning(s) ==="
fi

echo "=========================================="
echo ""
echo "Option A (File-Based):"
echo "  cd claude-multi-agent/option-a-file-based"
echo "  python coordination.py --help"
echo ""
echo "Option B (MCP Broker):"
echo "  cd claude-multi-agent/option-b-mcp-broker"
echo "  npm start"
echo ""
echo "Option C (Orchestrator):"
echo "  cd claude-multi-agent/option-c-orchestrator"
echo "  source .venv/bin/activate"
echo "  orchestrate --help"
echo ""
echo "Progress tracking:"
echo "  Progress file: $PROJECT_ROOT/claude-progress.txt"
echo "  Feature list:  $PROJECT_ROOT/feature_list.json"
echo ""
echo "For development:"
echo "  Read CODING_AGENT.md for session instructions"
echo ""
