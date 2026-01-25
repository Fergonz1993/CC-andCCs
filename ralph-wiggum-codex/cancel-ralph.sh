#!/bin/bash
#
# Cancel an active Ralph Wiggum loop
#

set -euo pipefail

STATE_FILE=".ralph-state.json"

if [[ -f "$STATE_FILE" ]]; then
    iteration=$(jq -r '.iteration' "$STATE_FILE" 2>/dev/null || echo "unknown")
    echo "Cancelling Ralph loop at iteration $iteration"
    rm "$STATE_FILE"
    echo "Ralph loop cancelled."
else
    echo "No active Ralph loop found."
fi
