#!/bin/bash
# Temporary background job: finish the Transfermarkt backfill in polite
# chunks, then remove itself. Run every 4 hours by
# scripts/com.felixnitschke.bundesliga-coach-impact.squadbackfill.plist.
#
# Each run is `bundesliga backfill`: one budget of at most 400 requests, 4 s
# apart, spent in priority order -- the Bundesliga coach list if it's over a
# week old, then coaching histories for the other top-5 leagues, then squads
# and injuries with whatever is left. Why chunks: Transfermarkt's firewall
# blocked the first full squad run after ~1,100 requests in an hour (see
# docs/engineering-notes.md). Every page is cached, so each run continues
# where the last one stopped; while Transfermarkt is still blocking, a run
# costs a single request and exits quietly.
#
# When everything is in (the backfill writes data/processed/backfill_complete
# only then), it notifies and unloads + deletes its own LaunchAgent -- this is
# not meant to become a standing job.
set -u

PROJECT_DIR=${PROJECT_DIR:-"/Users/felixnitschke/Felix.com/bundesliga-coach-impact"}
LABEL=${LABEL:-"com.felixnitschke.bundesliga-coach-impact.squadbackfill"}
PLIST=${PLIST:-"$HOME/Library/LaunchAgents/$LABEL.plist"}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-3600}
NOTIFY=${NOTIFY:-1}
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/squad_backfill_$(date +%Y-%m-%d_%H%M%S).log"
DONE_FILE="$PROJECT_DIR/data/processed/backfill_complete"

mkdir -p "$LOG_DIR"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"; }
notify() {
    [ "$NOTIFY" = "1" ] || return 0
    /usr/bin/osascript -e "display notification \"$2\" with title \"$1\"" >/dev/null 2>&1 || true
}
remove_self() {
    log "Removing the LaunchAgent ($LABEL) -- its work is done"
    rm -f "$PLIST"
    /bin/launchctl bootout "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true  # ends this job too; last step
}

cd "$PROJECT_DIR" || exit 1

if [ -f "$DONE_FILE" ]; then
    log "Backfill already complete"
    remove_self
    exit 0
fi
if pgrep -f "scripts/weekly_refresh.sh" >/dev/null; then
    log "Weekly refresh is running -- skipping this slot"
    exit 0
fi

source .venv/bin/activate
log "=== Backfill run (coach histories first, then squads) ==="
python cli.py backfill >> "$LOG_FILE" 2>&1 &
pid=$!
elapsed=0
while kill -0 "$pid" 2>/dev/null; do
    if [ "$elapsed" -ge "$TIMEOUT_SECONDS" ]; then
        log "Run exceeded ${TIMEOUT_SECONDS}s -- stopping it (progress is cached)"
        kill "$pid" 2>/dev/null
        break
    fi
    sleep 10
    elapsed=$((elapsed + 10))
done
wait "$pid"
status=$?

if [ -f "$DONE_FILE" ]; then
    log "Backfill complete"
    notify "Bundesliga backfill complete" "Coach histories for the top-5 leagues and the injury/signing data are in. Ask Claude to finish the analysis."
    remove_self
elif grep -q "blocked the scrape" "$LOG_FILE"; then
    log "Transfermarkt still blocking -- will try again next slot"
elif grep -q "request budget" "$LOG_FILE"; then
    log "Budget for this run used -- continuing next slot"
elif [ "$status" -ne 0 ]; then
    log "Run failed (exit $status)"
    notify "Bundesliga backfill error" "Exit $status. See $(basename "$LOG_FILE")"
fi

# keep the last ~30 run logs
ls -t "$LOG_DIR"/squad_backfill_*.log 2>/dev/null | tail -n +31 | while read -r old; do rm -f "$old"; done
exit 0
