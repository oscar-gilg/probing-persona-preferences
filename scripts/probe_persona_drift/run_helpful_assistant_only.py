"""Run only the helpful_assistant_{truth,harm} extractions.

Use case: filling in a control persona on a fresh pod without re-running the
14 existing extractions that are cached locally.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.probes.extraction import ExtractionConfig, run_extraction


CONFIG_DIR = Path("configs/extraction/probe_persona_drift")


def main():
    configs = sorted(CONFIG_DIR.glob("helpful_assistant_*.yaml"))
    print(f"Running {len(configs)} configs:")
    for cfg_path in configs:
        print(f"  - {cfg_path.name}")
    for i, cfg_path in enumerate(configs, 1):
        print(f"\n{'='*70}\n[{i}/{len(configs)}] {cfg_path.name}\n{'='*70}")
        cfg = ExtractionConfig.from_yaml(cfg_path, resume=True)
        run_extraction(cfg)
    print(f"\nDone.")


if __name__ == "__main__":
    main()
