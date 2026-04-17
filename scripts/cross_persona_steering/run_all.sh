#!/usr/bin/env bash
# Run all 4 cross-persona configs sequentially on a RunPod pod.
# Assumes the repo is at /workspace/repo, venv at /opt/venvs/research/bin/python, and .env present.
set -euo pipefail
cd /workspace/repo
set -a; source .env 2>/dev/null || true; set +a
PY=/opt/venvs/research/bin/python
for persona in sadist villain aesthete stem_obsessive; do
  echo "=================== $persona ==================="
  $PY -m src.steering.runner configs/steering/cross_persona/${persona}.yaml 2>&1 | tee /var/log/cps_${persona}.log
  echo "--- $persona DONE ---"
done
echo "ALL PERSONAS DONE"
