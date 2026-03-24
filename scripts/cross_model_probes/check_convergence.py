"""Check convergence status of measurement splits."""

import yaml
from pathlib import Path

base = Path("results/experiments/gptoss_qwen_mra2/pre_task_active_learning")
for d in sorted(base.iterdir()):
    model = d.name.split("_completion_")[0].replace("completion_preference_", "")
    split = d.name.split("split_")[1][0]

    # Check for final summary vs checkpoint
    summary_files = list(d.glob("active_learning_summary*.yaml"))
    checkpoint_files = list(d.glob("active_learning.yaml"))
    thu_files = list(d.glob("thurstonian_*.yaml"))

    n_scores = 0
    if thu_files:
        with open(thu_files[0]) as f:
            thu = yaml.safe_load(f)
        n_scores = len(thu.get("mu", {}))

    # Use checkpoint or summary
    src = summary_files[0] if summary_files else (checkpoint_files[0] if checkpoint_files else None)
    if not src:
        print(f"{model} split_{split}: NO DATA")
        continue
    with open(src) as f:
        s = yaml.safe_load(f)
    rc = s.get("rank_correlations", [])
    last_rc = rc[-1] if rc else 0
    status = "ABORTED" if s.get("aborted_api_failures") else ("CONVERGED" if s.get("converged") else "PARTIAL")
    print(
        f"{model} split_{split}: {status} "
        f"iters={s.get('n_iterations', '?')}, "
        f"comparisons={s.get('total_comparisons', 0)}, "
        f"rank_corr={last_rc:.4f}, "
        f"scores={n_scores}"
    )
