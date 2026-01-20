#!/bin/bash
# Worker Loop Script for File-Based Coordination
#
# This script continuously watches for available tasks and can be used
# to prompt Claude Code when new work is available.
#
# Usage: ./worker-loop.sh <terminal-id> [poll-interval-seconds]

TERMINAL_ID="${1:-worker-$(date +%s)}"
POLL_INTERVAL="${2:-5}"
COORDINATION_DIR=".coordination"
TASKS_FILE="$COORDINATION_DIR/tasks.json"

echo "=========================================="
echo "Worker Loop Started"
echo "Terminal ID: $TERMINAL_ID"
echo "Poll Interval: ${POLL_INTERVAL}s"
echo "=========================================="

# Check if coordination directory exists
if [ ! -d "$COORDINATION_DIR" ]; then
    echo "ERROR: Coordination directory not found. Run leader init first."
    exit 1
fi

# Function to check for available tasks
check_tasks() {
    if [ ! -f "$TASKS_FILE" ]; then
        return 1
    fi

    # Count available tasks with jq, default to 0 on failure
    AVAILABLE=$(jq '[.tasks[] | select(.status == "available")] | length' "$TASKS_FILE" 2>/dev/null || echo 0)
    # Ensure AVAILABLE is a valid integer
    if ! [[ "$AVAILABLE" =~ ^[0-9]+$ ]]; then
        AVAILABLE=0
    fi

    if [ "$AVAILABLE" -gt 0 ]; then
        echo ""
        echo "[$(date '+%H:%M:%S')] $AVAILABLE available task(s):"
        jq -r '.tasks[] | select(.status == "available") | "  [\(.priority)] \(.id): \(.description[:50])..."' "$TASKS_FILE" 2>/dev/null
        return 0
    fi
    return 1
}

# Function to show status summary
show_status() {
    if [ ! -f "$TASKS_FILE" ]; then
        return
    fi

    AVAILABLE=$(jq '[.tasks[] | select(.status == "available")] | length' "$TASKS_FILE" 2>/dev/null || echo 0)
    CLAIMED=$(jq '[.tasks[] | select(.status == "claimed")] | length' "$TASKS_FILE" 2>/dev/null || echo 0)
    IN_PROGRESS=$(jq '[.tasks[] | select(.status == "in_progress")] | length' "$TASKS_FILE" 2>/dev/null || echo 0)
    DONE=$(jq '[.tasks[] | select(.status == "done")] | length' "$TASKS_FILE" 2>/dev/null || echo 0)
    FAILED=$(jq '[.tasks[] | select(.status == "failed")] | length' "$TASKS_FILE" 2>/dev/null || echo 0)

    echo "[$(date '+%H:%M:%S')] Status: available=$AVAILABLE claimed=$CLAIMED in_progress=$IN_PROGRESS done=$DONE failed=$FAILED"
}

# Main loop
while true; do
    if check_tasks; then
        echo ""
        echo ">>> Tasks available! Run: python coordination.py worker claim $TERMINAL_ID"
        echo ""
    else
        show_status
    fi
    sleep "$POLL_INTERVAL"
done
