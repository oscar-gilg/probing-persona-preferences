"""Run all 14 probe-persona-drift extraction configs in a single process.

Loads the model once and iterates over configs, swapping system prompt + tasks
for each persona × target. Avoids paying the model-load cost (~1–2 min) per
extraction.

Usage:
    python -m scripts.probe_persona_drift.run_all_extractions

The configs live under configs/extraction/probe_persona_drift/ and are
processed in deterministic name order. If a particular extraction has already
written activations to its output_dir (resumable), the existing extraction's
internal resume logic skips completed items.
"""

from pathlib import Path

from src.probes.extraction import ExtractionConfig, run_extraction


CONFIG_DIR = Path("configs/extraction/probe_persona_drift")


def main():
    configs = sorted(CONFIG_DIR.glob("*.yaml"))
    print(f"Found {len(configs)} extraction configs.")
    for i, cfg_path in enumerate(configs, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(configs)}] Running {cfg_path.name}")
        print(f"{'='*70}")
        cfg = ExtractionConfig.from_yaml(cfg_path, resume=True)
        run_extraction(cfg)
    print(f"\nAll {len(configs)} extractions complete.")


if __name__ == "__main__":
    main()
