"""Batch-run all cross-model probe training configs.

Usage:
    python -m scripts.cross_model_probes.run_all_probes [--dry-run]
"""

import subprocess
import sys
from pathlib import Path

CONFIGS_DIR = Path("configs/probes/cross_model")


def main():
    dry_run = "--dry-run" in sys.argv

    configs = sorted(CONFIGS_DIR.glob("*.yaml"))
    print(f"Found {len(configs)} configs in {CONFIGS_DIR}")

    if dry_run:
        for c in configs:
            print(f"  {c.name}")
        return

    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(configs)}] {config.name}")
        print(f"{'='*60}")

        result = subprocess.run(
            [sys.executable, "-m", "src.probes.experiments.run_dir_probes", "--config", str(config)],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"FAILED: {config.name} (exit code {result.returncode})")


if __name__ == "__main__":
    main()
