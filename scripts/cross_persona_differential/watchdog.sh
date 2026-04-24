#!/usr/bin/env bash
# Fixed watchdog: only monitors the currently-parsing file, and only fires
# if the last log line matches "Parsing N completions" (we are in parse phase).
# Previous version incorrectly counted cross-persona totals, killing gen phases.
set -u
cd /workspace/repo
STALE_SECS="${STALE_SECS:-120}"

last_size=0
last_change=$(date +%s)
last_target=""

while true; do
    pid=$(pgrep -f "python -u -m src.steering.runner" | head -1)
    if [ -z "${pid:-}" ]; then
        sleep 15
        last_target=""
        continue
    fi

    # Only trigger if the log's recent tail shows we are actively parsing
    # (not in gen phase). "Parsing N completions" prints at parse start,
    # individual "[K/N]" lines during parse. If the *last* log line starts
    # with "  [" or "Parsing", we are parsing.
    last_line=$(tail -1 scripts/cross_persona_differential/sweep.log 2>/dev/null)
    case "$last_line" in
        "  ["* | "Parsing "*)
            in_parse=1 ;;
        *)
            in_parse=0 ;;
    esac

    if [ "$in_parse" -ne 1 ]; then
        sleep 15
        last_target=""
        continue
    fi

    # Find the most recently modified .parsed.jsonl
    target=$(ls -t experiments/cross_persona_differential/checkpoints/*.parsed.jsonl 2>/dev/null | head -1)
    if [ -z "${target:-}" ]; then
        sleep 15
        continue
    fi

    size=$(wc -l < "$target" 2>/dev/null || echo 0)
    now=$(date +%s)

    if [ "$target" != "$last_target" ] || [ "$size" != "$last_size" ]; then
        last_size=$size
        last_change=$now
        last_target=$target
    else
        stall=$((now - last_change))
        if [ "$stall" -ge "$STALE_SECS" ]; then
            echo "[$(date -u)] WATCHDOG: $target stuck at $size for ${stall}s → SIGKILL pid=$pid"
            kill -9 "$pid" 2>&1
            sleep 30
            last_change=$(date +%s)
        fi
    fi

    sleep 15
done
