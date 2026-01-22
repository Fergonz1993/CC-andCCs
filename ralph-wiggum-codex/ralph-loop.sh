#!/bin/bash
#
# Ralph Wiggum Loop for OpenAI Codex CLI
# "Deterministically bad in an undeterministic world"
#
# Prerequisites:
#   npm i -g @openai/codex
#   Then run: codex (and sign in with ChatGPT)
#
# Usage: ./ralph-loop.sh "Your task here" [OPTIONS]

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Defaults
MAX_ITERATIONS=50
COMPLETION_PROMISE="TASK COMPLETE"
WORKING_DIR="."
STATE_FILE=".ralph-state.json"
LOG_DIR=".ralph-logs"
OUTPUT_FILE=".ralph-last-output.txt"
FULL_AUTO=false
YOLO_MODE=false

# Help
show_help() {
    cat << EOF
Ralph Wiggum Loop for OpenAI Codex CLI

Usage: $0 "PROMPT" [OPTIONS]

Arguments:
  PROMPT                    The task/prompt to run repeatedly

Options:
  --max-iterations N        Max iterations before stopping (default: 50)
  --completion-promise TXT  Text to detect completion (default: "TASK COMPLETE")
  --working-dir DIR         Working directory (default: current)
  --full-auto               Run with --full-auto flag (auto-approve safe operations)
  --yolo                    Run with --yolo flag (DANGEROUS: no sandbox, no approvals)
  -h, --help                Show this help

Examples:
  $0 "Add unit tests for utils.js" --max-iterations 20
  $0 "Fix the auth bug" --full-auto --completion-promise "BUG FIXED"

The AI must output <promise>TASK COMPLETE</promise> to signal completion.
EOF
    exit 0
}

# Parse arguments
PROMPT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --completion-promise)
            COMPLETION_PROMISE="$2"
            shift 2
            ;;
        --working-dir)
            WORKING_DIR="$2"
            shift 2
            ;;
        --full-auto)
            FULL_AUTO=true
            shift
            ;;
        --yolo)
            YOLO_MODE=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
        *)
            if [[ -z "$PROMPT" ]]; then
                PROMPT="$1"
            else
                PROMPT="$PROMPT $1"
            fi
            shift
            ;;
    esac
done

if [[ -z "$PROMPT" ]]; then
    echo -e "${RED}Error: No prompt provided${NC}"
    echo "Usage: $0 \"Your task here\" [OPTIONS]"
    exit 1
fi

# Check codex is installed
if ! command -v codex &> /dev/null; then
    echo -e "${RED}Error: codex CLI not found${NC}"
    echo "Install with: npm i -g @openai/codex"
    echo "Then run: codex (to sign in)"
    exit 1
fi

cd "$WORKING_DIR"
mkdir -p "$LOG_DIR"

# Build the full prompt with Ralph instructions
build_prompt() {
    cat << EOF
You are in a RALPH WIGGUM LOOP. This exact prompt will be fed to you repeatedly until you complete the task.

## YOUR TASK
$PROMPT

## CRITICAL RULES

1. **CHECK YOUR PREVIOUS WORK FIRST**
   - Look at files you may have already modified
   - Check git status and git log for your previous commits
   - Read $STATE_FILE to see what iteration you're on
   - DON'T start over - build on what exists

2. **MAKE INCREMENTAL PROGRESS**
   - Each iteration should move closer to completion
   - If something failed last time, try a different approach
   - Read error messages and fix them

3. **SIGNAL COMPLETION**
   When the task is FULLY COMPLETE and VERIFIED:
   Output exactly: <promise>$COMPLETION_PROMISE</promise>

   DO NOT output this until you have:
   - Implemented the solution
   - Tested/verified it works
   - Confirmed all requirements are met

## ITERATION INFO
Check $STATE_FILE for current iteration count.
Check $LOG_DIR/ for logs of previous iterations.

## REMEMBER
- You will keep getting this same prompt until you output the completion promise
- Your file changes persist between iterations
- Use git commits to track your progress
- If stuck, try something different than last time
EOF
}

# Initialize state
init_state() {
    if [[ -f "$STATE_FILE" ]]; then
        # Check if jq is available before using it
        if command -v jq &> /dev/null; then
            ITERATION=$(jq -r '.iteration // 0' "$STATE_FILE" 2>/dev/null || echo 0)
        else
            # Fallback: try to extract iteration with grep/sed if jq is missing
            ITERATION=$(grep -o '"iteration"[[:space:]]*:[[:space:]]*[0-9]*' "$STATE_FILE" 2>/dev/null | grep -o '[0-9]*' || echo 0)
        fi
    else
        ITERATION=0
        echo '{"iteration": 0, "status": "starting"}' > "$STATE_FILE"
    fi
}

# Update state
update_state() {
    local status="$1"
    local now
    now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    if command -v jq &> /dev/null; then
        echo "{\"iteration\": $ITERATION, \"status\": \"$status\", \"last_update\": \"$now\", \"prompt\": $(echo "$PROMPT" | jq -Rs .)}" > "$STATE_FILE"
    else
        echo "{\"iteration\": $ITERATION, \"status\": \"$status\", \"last_update\": \"$now\"}" > "$STATE_FILE"
    fi
}

# Check for completion
check_completion() {
    local output_file="$1"
    # Use fixed-string matching to avoid regex injection from COMPLETION_PROMISE
    # First check if <promise> tags exist, then check for the literal promise text
    if grep -F "<promise>" "$output_file" 2>/dev/null | grep -F "$COMPLETION_PROMISE" | grep -qF "</promise>"; then
        return 0
    fi
    # Alternative: check if the exact promise appears between tags on a single line
    if grep -F "<promise>${COMPLETION_PROMISE}</promise>" "$output_file" 2>/dev/null; then
        return 0
    fi
    return 1
}

# Run one iteration
run_iteration() {
    local log_file="${LOG_DIR}/iteration-${ITERATION}.log"
    local full_prompt
    full_prompt=$(build_prompt)

    echo -e "${BLUE}[Ralph]${NC} Iteration ${CYAN}$ITERATION${NC} / $MAX_ITERATIONS"

    # Build codex command
    local codex_args=("exec")

    if $FULL_AUTO; then
        codex_args+=("--full-auto")
        echo -e "${YELLOW}[Ralph]${NC} Running with --full-auto"
    fi

    if $YOLO_MODE; then
        codex_args+=("--yolo")
        echo -e "${RED}[Ralph]${NC} Running with --yolo (DANGEROUS)"
    fi

    codex_args+=("--output-last-message" "$OUTPUT_FILE")

    echo -e "${BLUE}[Ralph]${NC} Executing codex..."

    # Run codex with the prompt and capture its exit status from PIPESTATUS
    echo "$full_prompt" | codex "${codex_args[@]}" 2>&1 | tee "$log_file"
    local codex_exit_code=${PIPESTATUS[1]}

    if [[ "$codex_exit_code" -eq 0 ]]; then
        echo -e "${GREEN}[Ralph]${NC} Iteration $ITERATION completed"
    else
        echo -e "${YELLOW}[Ralph]${NC} Iteration $ITERATION had errors (exit code: $codex_exit_code)"
    fi

    # Also check the output file if it exists
    if [[ -f "$OUTPUT_FILE" ]]; then
        cat "$OUTPUT_FILE" >> "$log_file"
    fi
}

# Main loop
main() {
    echo -e "${GREEN}"
    cat << 'EOF'
╔═══════════════════════════════════════════════════════════╗
║         RALPH WIGGUM LOOP - CODEX CLI EDITION             ║
║        "I'm in danger... of iterating forever"            ║
╚═══════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"

    echo -e "${BLUE}[Config]${NC} Max iterations: $MAX_ITERATIONS"
    echo -e "${BLUE}[Config]${NC} Completion promise: '${CYAN}$COMPLETION_PROMISE${NC}'"
    echo -e "${BLUE}[Config]${NC} Working dir: $WORKING_DIR"
    echo -e "${BLUE}[Config]${NC} Prompt: ${CYAN}${PROMPT:0:60}...${NC}"
    echo ""

    init_state

    while [[ $ITERATION -lt $MAX_ITERATIONS ]]; do
        ((ITERATION++))
        update_state "running"

        local log_file="${LOG_DIR}/iteration-${ITERATION}.log"

        run_iteration

        # Check for completion in log and output file
        if check_completion "$log_file" || check_completion "$OUTPUT_FILE" 2>/dev/null; then
            echo -e "${GREEN}"
            cat << 'EOF'
╔═══════════════════════════════════════════════════════════╗
║                    TASK COMPLETE!                         ║
║          Ralph has escaped the loop (for now)             ║
╚═══════════════════════════════════════════════════════════╝
EOF
            echo -e "${NC}"
            update_state "completed"
            echo -e "${BLUE}[Ralph]${NC} Completed in $ITERATION iterations"
            exit 0
        fi

        echo -e "${YELLOW}[Ralph]${NC} No completion promise detected. Continuing loop..."
        echo ""
        sleep 2
    done

    echo -e "${RED}"
    cat << 'EOF'
╔═══════════════════════════════════════════════════════════╗
║               MAX ITERATIONS REACHED                      ║
║           Ralph is tired. Task incomplete.                ║
╚═══════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    update_state "max_iterations"
    exit 1
}

# Cleanup on interrupt
cleanup() {
    echo -e "\n${YELLOW}[Ralph]${NC} Interrupted at iteration $ITERATION"
    update_state "interrupted"
    exit 130
}

trap cleanup SIGINT SIGTERM

main
