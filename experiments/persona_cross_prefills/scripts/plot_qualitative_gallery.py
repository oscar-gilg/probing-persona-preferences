"""Qualitative per-token heatmaps. Reuses scripts/distress/per_token_analysis.py::render_anthropic_heatmap.

Two galleries:
- Representative: 1 prefill per (condition × persona) cell, the one with user-EOT score closest to the cell mean.
- Surprising: top-2 prefills with largest deviation from cell mean (per condition × persona).

Each prefill is rendered as one heatmap PNG; we also stitch a contact-sheet PDF per gallery.

Usage: python -m experiments.persona_cross_prefills.scripts.plot_qualitative_gallery
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.distress.per_token_analysis import render_anthropic_heatmap  # noqa: E402

EXP = ROOT / "experiments/persona_cross_prefills"
RESULTS = EXP / "results"
ASSETS = EXP / "assets"
GALLERY_DIR = ASSETS / "gallery"
TODAY = date.today().strftime("%m%d%y")

CONDITION_ORDER = ["benign_helpful", "benign_evil", "harmful_refused", "harmful_obliged"]
PERSONAS = ["default", "sadist"]


def load_personas() -> dict[str, str]:
    return json.loads((EXP / "personas.json").read_text())["personas"]


def load_prefill_messages() -> dict[str, list[dict]]:
    """Map prefill_id → original messages (no system prompt)."""
    out = {}
    for fname in ["prefills_benign.json", "prefills_harmful.json"]:
        items = json.loads((EXP / fname).read_text())
        for item in items:
            out[item["prefill_id"]] = item["messages"]
    return out


def reconstruct_messages(prefill_messages: list[dict], persona_text: str) -> list[dict]:
    """Mirror the scoring-time persona injection."""
    if persona_text:
        return [{"role": "system", "content": persona_text}, *prefill_messages]
    return list(prefill_messages)


def pick_representative(items: list[dict]) -> dict:
    user_eots = np.array([it["eot_scores"]["user_eot"] for it in items])
    target = user_eots.mean()
    idx = int(np.argmin(np.abs(user_eots - target)))
    return items[idx]


def pick_surprising(items: list[dict], k: int = 2) -> list[dict]:
    user_eots = np.array([it["eot_scores"]["user_eot"] for it in items])
    target = user_eots.mean()
    diffs = np.abs(user_eots - target)
    top_k = np.argsort(diffs)[-k:][::-1]
    return [items[i] for i in top_k]


def render_one(item: dict, prefill_messages: list[dict], persona_text: str, tokenizer, out_dir: Path) -> Path:
    messages = reconstruct_messages(prefill_messages, persona_text)
    scores = np.array(item["per_token_scores"], dtype=np.float32)
    title = (f"{item['prefill_id']} | persona={item['persona_name']} | "
             f"asst-EOT={item['eot_scores']['asst_eot']:.2f} | "
             f"user-EOT={item['eot_scores']['user_eot']:.2f}")
    out = out_dir / f"{item['prefill_id']}__{item['persona_name']}.png"
    render_anthropic_heatmap(
        tokenizer=tokenizer,
        messages=messages,
        token_scores=scores,
        title=title,
        out=out,
        max_turns=4 if persona_text else 3,
        tokens_per_line=22,
    )
    return out


def main():
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    rep_dir = GALLERY_DIR / "representative"
    sur_dir = GALLERY_DIR / "surprising"
    rep_dir.mkdir(exist_ok=True)
    sur_dir.mkdir(exist_ok=True)

    print("Loading scoring results, prefills, personas, tokenizer...")
    data = json.loads((RESULTS / "scoring_results.json").read_text())
    items_by_cell: dict[tuple[str, str], list[dict]] = {}
    for item in data["items"]:
        key = (item["condition"], item["persona_name"])
        items_by_cell.setdefault(key, []).append(item)

    prefill_messages = load_prefill_messages()
    personas = load_personas()
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

    rep_paths = []
    sur_paths = []
    for cond in CONDITION_ORDER:
        for persona in PERSONAS:
            cell = items_by_cell[(cond, persona)]
            persona_text = personas[persona]

            rep = pick_representative(cell)
            rep_paths.append(render_one(rep, prefill_messages[rep["prefill_id"]], persona_text, tokenizer, rep_dir))
            print(f"  representative {cond}/{persona}: {rep['prefill_id']} → {rep_paths[-1].name}")

            for sur in pick_surprising(cell, k=2):
                sur_paths.append(render_one(sur, prefill_messages[sur["prefill_id"]], persona_text, tokenizer, sur_dir))
                print(f"  surprising  {cond}/{persona}: {sur['prefill_id']} → {sur_paths[-1].name}")

    print(f"\nRendered {len(rep_paths)} representative + {len(sur_paths)} surprising heatmaps")
    print(f"  in {GALLERY_DIR}/")


if __name__ == "__main__":
    main()
