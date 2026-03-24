#!/bin/bash
cd /workspace/repo
for cfg in configs/probes/cross_model/qwen35_acts_*.yaml; do
    echo "Training: $cfg"
    /workspace/venv/bin/python -m src.probes.experiments.run_dir_probes --config "$cfg"
done
echo "Probes done, running cross-eval"
/workspace/venv/bin/python -m scripts.cross_model_probes.cross_eval
