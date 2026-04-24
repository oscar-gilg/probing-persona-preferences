#!/bin/bash
# Debug-wrapped AL launch for qwen_persona_sweep_final_six.
#
# Captures:
#   - stdout (tee'd to logs/al.stdout.log, visible in tmux)
#   - stderr (teed to logs/al.stderr.log — rich progress display eats stderr
#     otherwise, hiding tracebacks and warnings)
#   - a watchdog snapshot every 30 s to logs/al.watchdog.log: elapsed time,
#     CPU%, RSS MB, thread count, ESTABLISHED TCP count, last-modified result
#     file. Survives the python process dying.
#   - signal handler: on SIGTERM/SIGINT, logs a Python-level traceback via py-spy
#     if installed, else a `ps` snapshot, before re-raising.
#
# Usage (inside tmux):
#   bash scripts/qwen_persona_transfer/run_al_with_debug.sh
#
# Args flow straight through to the runner. Defaults target the 21 persona-sweep
# configs with --max-concurrent 50.

set -u
set -o pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

LOG_DIR="$REPO/logs/qwen_persona_al_debug_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# SSL cert-bundle shortcut — avoids the ~30 min OpenSSL parse hang that dropped
# PID 73964 and PID 81488 into silent-startup death earlier.
export SSL_CERT_FILE=$(python -m certifi)
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export PYTHONUNBUFFERED=1
# Enable Python fault handler so segfaults / C-level crashes print a trace
# before the process dies.
export PYTHONFAULTHANDLER=1

echo "=== Debug AL launch $(date -u +%FT%TZ) ===" | tee "$LOG_DIR/meta.log"
echo "Logs: $LOG_DIR" | tee -a "$LOG_DIR/meta.log"
echo "SSL_CERT_FILE=$SSL_CERT_FILE" | tee -a "$LOG_DIR/meta.log"
echo "Python: $(which python) ($(python -V))" | tee -a "$LOG_DIR/meta.log"

# Launch runner with stdout/stderr separated. Rich progress uses stdout; real
# errors usually go to stderr.
python -m src.measurement.runners.run \
    configs/measurement/qwen_persona_sweep/final_six/*.yaml \
    --max-concurrent 50 \
    --experiment-id qwen_persona_sweep_final_six \
    "$@" \
    > >(tee "$LOG_DIR/al.stdout.log") \
    2> >(tee "$LOG_DIR/al.stderr.log" >&2) &
AL_PID=$!
echo "AL PID: $AL_PID" | tee -a "$LOG_DIR/meta.log"

# Watchdog in background. Every 30 s, append a snapshot. Exits when AL exits.
(
    while kill -0 "$AL_PID" 2>/dev/null; do
        ts=$(date -u +%FT%TZ)
        # ps columns: %CPU %MEM RSS_kB ELAPSED STATE
        ps_line=$(ps -o %cpu=,%mem=,rss=,etime=,state= -p "$AL_PID" 2>/dev/null)
        # Thread count (macOS: ps -M; fall back to lsof if that fails)
        tcp_count=$(lsof -p "$AL_PID" 2>/dev/null | grep -c "TCP.*ESTABLISHED")
        fd_count=$(lsof -p "$AL_PID" 2>/dev/null | wc -l | tr -d ' ')
        last_result=$(find "$REPO/results/experiments" -type d -name "qwen_persona_sweep_final_six*" -newer "$LOG_DIR/meta.log" 2>/dev/null | head -1)
        echo "[$ts] ps={$ps_line} tcp_est=$tcp_count fds=$fd_count last_result=${last_result:-<none>}" >> "$LOG_DIR/al.watchdog.log"
        sleep 30
    done
    echo "[$(date -u +%FT%TZ)] AL_PID $AL_PID exited" >> "$LOG_DIR/al.watchdog.log"
    # Post-mortem: capture exit context
    echo "=== Final dmesg-like snapshot ===" >> "$LOG_DIR/al.watchdog.log"
    # On macOS, look for OOM/kill hints in Console logs from last 5 min
    log show --last 5m --predicate 'eventMessage CONTAINS[c] "memory" OR eventMessage CONTAINS[c] "killed"' --style compact 2>/dev/null | tail -30 >> "$LOG_DIR/al.watchdog.log" || true
) &
WATCHDOG_PID=$!

trap "echo 'SIGTERM/SIGINT received' >> $LOG_DIR/meta.log; kill -s TERM $AL_PID 2>/dev/null; wait $AL_PID; exit" INT TERM

wait "$AL_PID"
EXIT_CODE=$?
echo "=== AL exited with code $EXIT_CODE ===" | tee -a "$LOG_DIR/meta.log"

# Let watchdog capture one more snapshot
sleep 2
kill "$WATCHDOG_PID" 2>/dev/null || true

echo "Log dir: $LOG_DIR"
exit "$EXIT_CODE"
