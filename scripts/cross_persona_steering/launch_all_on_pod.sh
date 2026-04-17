#!/usr/bin/env bash
# Launch all 4 persona configs in separate tmux sessions on a RunPod pod.
# Run remotely via: ssh runpod-cross_persona_steering 'bash -s' < scripts/cross_persona_steering/launch_all_on_pod.sh
# Or: scp this script to the pod and run there.
#
# Assumes:
#   - cwd on pod is /workspace/repo
#   - Python at /opt/venvs/research/bin/python
#   - tmux installed
#   - .env present at /workspace/repo/.env

set -euo pipefail

cd /workspace/repo
PY=/opt/venvs/research/bin/python
LOG_DIR=/var/log
mkdir -p "$LOG_DIR"

for persona in sadist villain aesthete stem_obsessive; do
  session="cps_${persona}"
  config="configs/steering/cross_persona/${persona}.yaml"
  log="${LOG_DIR}/cps_${persona}.log"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Session $session already exists; skipping (tmux kill-session -t $session to restart)."
    continue
  fi

  echo "Launching $session -> $log"
  tmux new-session -d -s "$session" \
    "$PY -m src.steering.runner $config 2>&1 | tee $log; echo DONE; sleep 3600"
done

echo
echo "Active tmux sessions:"
tmux list-sessions
