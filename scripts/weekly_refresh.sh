#!/bin/bash
# Weekly data refresh for the Bundesliga coach impact pipeline, run by
# launchd (see scripts/com.felixnitschke.bundesliga-coach-impact.weeklyrefresh.plist
# and README "Automated weekly refresh").
#
# launchd invokes this with almost no environment (no PATH beyond
# /usr/bin:/bin, no shell profile sourced) -- everything here uses absolute
# paths or macOS built-ins for exactly that reason.
#
# Hardened after the first two scheduled runs both failed (Sept 2026):
#   - One hung for 29 hours inside the FBref/Selenium fetch, with nothing
#     to stop it -> every attempt now runs under a watchdog that kills the
#     whole process group (Python AND the Chrome/chromedriver it spawned)
#     after ATTEMPT_TIMEOUT_SECONDS. macOS ships no `timeout` command, so
#     the watchdog is plain bash.
#   - The other ran the instant the Mac woke, before Wi-Fi was back, and
#     died on a DNS error -> the script now waits for the data sources to
#     be reachable before starting, and retries a failed attempt.
#   - Neither failure told anyone -> a failure now raises a macOS
#     notification, not just a line in a log file.
set -u

PROJECT_DIR=${PROJECT_DIR:-"/Users/felixnitschke/Felix.com/bundesliga-coach-impact"}
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/refresh_$(date +%Y-%m-%d_%H%M%S).log"
STATUS_FILE="$LOG_DIR/last_refresh_status"

NETWORK_WAIT_SECONDS=${NETWORK_WAIT_SECONDS:-900}       # give Wi-Fi up to 15 min after wake
ATTEMPT_TIMEOUT_SECONDS=${ATTEMPT_TIMEOUT_SECONDS:-2700} # a healthy run takes ~6 min
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
RETRY_DELAY_SECONDS=${RETRY_DELAY_SECONDS:-600}
PROBE_URL=${PROBE_URL:-"https://understat.com"}
NOTIFY=${NOTIFY:-1}

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

notify() {
    [ "$NOTIFY" = "1" ] || return 0
    # Works because the LaunchAgent runs inside the user's GUI session.
    /usr/bin/osascript -e "display notification \"$2\" with title \"$1\"" >/dev/null 2>&1 || true
}

wait_for_network() {
    local waited=0
    until /usr/bin/curl -sfI --max-time 10 "$PROBE_URL" >/dev/null 2>&1; do
        if [ "$waited" -ge "$NETWORK_WAIT_SECONDS" ]; then
            log "Network still unreachable after ${NETWORK_WAIT_SECONDS}s ($PROBE_URL)"
            return 1
        fi
        sleep 30
        waited=$((waited + 30))
    done
    log "Network reachable (waited ${waited}s)"
}

# Run "$@" in its own process group; kill the whole group if it outlives
# $1 seconds. Returns the command's exit status, or 124 on timeout (the
# same convention as GNU timeout).
run_with_timeout() {
    local limit=$1; shift
    set -m                       # job control -> the background job gets its own process group
    "$@" &
    local pid=$!
    set +m
    local elapsed=0
    while kill -0 "$pid" 2>/dev/null; do
        if [ "$elapsed" -ge "$limit" ]; then
            log "Attempt exceeded ${limit}s -- killing process group $pid (incl. Chrome/chromedriver)"
            kill -TERM -- "-$pid" 2>/dev/null
            sleep 10
            kill -KILL -- "-$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
            return 124
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    wait "$pid"
}

STATUS=1
{
    log "=== Weekly refresh started ==="
    cd "$PROJECT_DIR" && source .venv/bin/activate

    if wait_for_network; then
        for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
            log "--- Attempt $attempt/$MAX_ATTEMPTS ---"
            run_with_timeout "$ATTEMPT_TIMEOUT_SECONDS" python cli.py pipeline
            STATUS=$?
            [ "$STATUS" -eq 0 ] && break
            log "Attempt $attempt failed (exit $STATUS)"
            if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
                sleep "$RETRY_DELAY_SECONDS"
                wait_for_network || break
            fi
        done
    fi

    if [ "$STATUS" -eq 0 ]; then
        log "=== Weekly refresh finished OK ==="
    else
        log "=== Weekly refresh FAILED (exit $STATUS) ==="
    fi
} >> "$LOG_FILE" 2>&1

if [ "$STATUS" -eq 0 ]; then
    echo "ok $(date '+%Y-%m-%dT%H:%M:%S') $LOG_FILE" > "$STATUS_FILE"
else
    echo "failed $(date '+%Y-%m-%dT%H:%M:%S') exit=$STATUS $LOG_FILE" > "$STATUS_FILE"
    notify "Bundesliga refresh failed" "Exit $STATUS after $MAX_ATTEMPTS attempts. See $(basename "$LOG_FILE")"
fi

# Keep the last ~12 logs (roughly 3 months at weekly cadence). Portable
# (no GNU-only xargs -r) since this needs to work with macOS's BSD userland.
ls -t "$LOG_DIR"/refresh_*.log 2>/dev/null | tail -n +13 | while read -r old_log; do
    rm -f "$old_log"
done

exit "$STATUS"
