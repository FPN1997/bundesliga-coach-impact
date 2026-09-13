#!/bin/bash
# Weekly data refresh for the Bundesliga coach impact pipeline, run by cron.
#
# Cron invokes this with almost no environment (no PATH beyond /usr/bin:/bin,
# no shell profile sourced) -- everything here uses absolute paths for
# exactly that reason. `source .venv/bin/activate` still works fine since
# `source` is a shell builtin, not something looked up on PATH.
#
# See README.md's "Automated weekly refresh" section for setup, the macOS
# Full Disk Access requirement, and known limitations (a sleeping Mac simply
# skips that week's run -- cron does not queue or catch up missed jobs).
set -u

PROJECT_DIR="/Users/felixnitschke/Felix.com/bundesliga-coach-impact"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/refresh_$(date +%Y-%m-%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

{
    echo "=== Weekly refresh started: $(date) ==="
    cd "$PROJECT_DIR" && source .venv/bin/activate && python run_pipeline.py
    STATUS=$?
    if [ "$STATUS" -eq 0 ]; then
        echo "=== Weekly refresh finished OK: $(date) ==="
    else
        echo "=== Weekly refresh FAILED (exit $STATUS): $(date) ==="
    fi
} >> "$LOG_FILE" 2>&1

# Keep the last ~12 logs (roughly 3 months at weekly cadence) regardless of
# success/failure, so this doesn't grow forever. Portable (no GNU-only
# xargs -r) since this needs to work with macOS's BSD userland.
ls -t "$LOG_DIR"/refresh_*.log 2>/dev/null | tail -n +13 | while read -r old_log; do
    rm -f "$old_log"
done

exit "$STATUS"
