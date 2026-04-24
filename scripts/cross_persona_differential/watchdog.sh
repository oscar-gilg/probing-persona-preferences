#!/usr/bin/env bash
# Watchdog: if any .parsed.jsonl file stops growing for >90s while the runner
# python process exists, SIGKILL the runner so the robust wrapper retries.
# Only monitors personas that are actively parsing (not in gen phase).
set -u
cd /workspace/repo
STALE_SECS="${STALE_SECS:-120}"

last_size=""
last_change=$(date +%s)

while true; do
    # Find the current python runner PID (if any)
    pid=$(pgrep -f "python -u -m src.steering.runner" | head -1)
    if [ -z "${pid:-}" ]; then
        sleep 15
        continue
    fi

    # Sum rows across all parsed files (parse progress signal).
    # Gen progress can also be used but gen rarely hangs.
    total=$(cat experiments/cross_persona_differential/checkpoints/*.parsed.jsonl 2>/dev/null | wc -l || echo 0)

    now=$(date +%s)
    if [ "$total" = "$last_size" ]; then
        # Also check: are we in parse phase? Look for recent "Parsing" in log.
        if grep -q "Parsing [0-9]* completions" scripts/cross_persona_differential/sweep.log 2>/dev/null; then
            stall=$((now - last_change))
            if [ "$stall" -ge "$STALE_SECS" ]; then
                echo "[$(date -u)] WATCHDOG: parse total stuck at $total for ${stall}s → SIGKILL pid=$pid"
                kill -9 "$pid" 2>&1
                last_change=$now
                sleep 20
                continue
            fi
        fi
    else
        last_size=$total
        last_change=$now
    fi

    sleep 15
done
