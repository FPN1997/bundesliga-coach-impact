#!/bin/bash
# Rebuild the results page and publish it: commit site/ and push to main,
# which triggers .github/workflows/pages.yml. Called by weekly_refresh.sh
# after a successful data refresh; safe to run by hand too.
#
# Deliberately conservative, since it pushes to a public repo unattended:
#   - publishes only when the page has newer match data than the copy
#     already on main (no empty weekly commits in the summer break);
#   - only from the main branch, and never when origin/main has commits
#     that aren't here (it won't merge or rebase on its own);
#   - commits ONLY site/ -- anything else in the working tree, staged or
#     not, is left exactly as it was.
# The odds refresh is allowed to fail (the page then keeps the previous
# benchmark numbers); the page build is not.
#
# Exit codes: 0 published or nothing new to publish; 1 anything else.
set -u

PROJECT_DIR=${PROJECT_DIR:-"/Users/felixnitschke/Felix.com/bundesliga-coach-impact"}
BRANCH=main
GIT=/usr/bin/git

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] publish: $*"; }
data_through() { grep -o '"data_through":"[0-9-]*"' | head -1 | grep -o '[0-9]\{4\}-[0-9-]*'; }

cd "$PROJECT_DIR" || exit 1
[ -f .venv/bin/activate ] && source .venv/bin/activate

current=$($GIT symbolic-ref --short HEAD 2>/dev/null)
if [ "$current" != "$BRANCH" ]; then
    log "on branch '$current', not $BRANCH -- not publishing"
    exit 1
fi

if ! python cli.py benchmark --refresh-odds; then
    log "benchmark/odds refresh failed -- continuing with the previous benchmark results"
fi
python cli.py site || { log "page build failed"; exit 1; }

new=$(data_through < site/index.html)
old=$($GIT show "HEAD:site/index.html" 2>/dev/null | data_through)
if [ -n "$new" ] && [ "$new" = "$old" ]; then
    log "no new matches since the published page (data through $new) -- nothing to publish"
    $GIT checkout -- site/ 2>/dev/null   # drop the rebuilt-but-unchanged page
    exit 0
fi

if ! $GIT fetch -q origin "$BRANCH"; then
    log "git fetch failed -- not publishing"
    exit 1
fi
if ! $GIT merge-base --is-ancestor "origin/$BRANCH" HEAD; then
    log "origin/$BRANCH has commits that aren't in this checkout -- pull first; not publishing"
    exit 1
fi

$GIT add -- site/
if $GIT diff --cached --quiet -- site/; then
    log "page unchanged -- nothing to publish"
    exit 0
fi
$GIT commit -q -m "Weekly results-page update: data through ${new}" -- site/ || { log "commit failed"; exit 1; }
if ! $GIT push -q origin "HEAD:$BRANCH"; then
    log "push failed -- the commit stays local and goes out with the next successful push"
    exit 1
fi
log "published page with data through $new ($($GIT rev-parse --short HEAD))"
