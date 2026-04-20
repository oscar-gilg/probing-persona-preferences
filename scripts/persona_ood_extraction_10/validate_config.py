"""Dry-run parse + task-loading for one persona config."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from src.probes.extraction import ExtractionConfig
from src.probes.extraction.extract import _load_tasks


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/extraction/pref_evil_genius.yaml")
    cfg = ExtractionConfig.from_yaml(config_path)
    print(f"parsed OK: model={cfg.model} output={cfg.output_dir}")
    print(f"selectors={cfg.selectors}")
    print(f"layers={cfg.layers_to_extract}")
    tasks = _load_tasks(cfg)
    c = Counter(t.origin.name for t in tasks)
    print(f"loaded {len(tasks)} tasks, origins: {dict(c)}")
    with open("data/canonical_splits/test_task_ids.txt") as f:
        expected = {l.strip() for l in f if l.strip()}
    loaded = {t.id for t in tasks}
    missing = expected - loaded
    extra = loaded - expected
    print(f"expected {len(expected)}; missing from load={len(missing)} extra={len(extra)}")
    if missing:
        for m in list(missing)[:5]:
            print(f"  missing example: {m}")


if __name__ == "__main__":
    main()
