"""Recompute the §3.1 structured claims from the v2 eot_discrimination data.

Replaces the v1 values in
`paper/claims/canonical_probe_eval_make_paper_figures.json` for:
  - "Persona modulation d full" (all (domain, sysprompt) cells, Gemma)
  - "Per turn parent absolute d" (parent task-averaged probe, both turns)
  - "Per turn absolute d truth"   (per-(turn, probe-family))
  - "Per turn absolute d harm"    (per-(turn, probe-family))

Scalar claims for base discrimination (already updated by hand) are left alone;
this script only touches the four structured claims listed above.

Usage:
    python -m scripts.paper.refresh_eot_v2_claims
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
V2_SCORING = REPO_ROOT / "experiments/eot_discrimination_v2/scoring/gemma3_27b"
USER_TURN_PATH = V2_SCORING / "user_turn_scoring_results.json"
ASSISTANT_TURN_PATH = V2_SCORING / "scoring_results.json"
CLAIMS_PATH = REPO_ROOT / "paper/claims/canonical_probe_eval_make_paper_figures.json"

DOMAIN_PROBE = {"truth": "tb-5_L32", "harm": "tb-5_L39", "politics": "tb-5_L39"}
DOMAIN_PARENT = {"truth": "task_mean_L32", "harm": "task_mean_L39"}
DOMAIN_POS = {"truth": "true", "harm": "harmful", "politics": "left"}
DOMAIN_NEG = {"truth": "false", "harm": "benign", "politics": "right"}

PERSONA_MODULATION_ORDER = {
    "truth": ["truthful", "neutral", "unreliable_narrator", "contrarian",
              "opposite_day", "lie_directive", "pathological_liar",
              "con_artist", "gaslighter"],
    "harm": ["safe", "neutral", "unrestricted", "sinister_ai", "sadist"],
    "politics": ["socialist", "democrat", "centrist", "apolitical", "neutral",
                 "libertarian", "republican", "nationalist", "contrarian"],
}


def cohen_d_pooled(pos: list[float], neg: list[float]) -> float:
    pos_arr = np.asarray(pos, float)
    neg_arr = np.asarray(neg, float)
    if len(pos_arr) < 2 or len(neg_arr) < 2:
        return float("nan")
    pooled = np.sqrt(
        ((len(pos_arr) - 1) * pos_arr.var(ddof=1)
         + (len(neg_arr) - 1) * neg_arr.var(ddof=1))
        / (len(pos_arr) + len(neg_arr) - 2)
    )
    if pooled == 0:
        return 0.0
    return float((pos_arr.mean() - neg_arr.mean()) / pooled)


def load_items(path: Path) -> list[dict]:
    return json.loads(path.read_text())["items"]


def compute_persona_modulation_d_full(assistant_items: list[dict]) -> dict:
    by_domain_sp: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for it in assistant_items:
        by_domain_sp[it["domain"]][it["system_prompt"]].append(it)

    table: dict[str, dict[str, float]] = {}
    for domain, order in PERSONA_MODULATION_ORDER.items():
        probe = DOMAIN_PROBE[domain]
        c_pos, c_neg = DOMAIN_POS[domain], DOMAIN_NEG[domain]
        row: dict[str, float] = {}
        for sp in order:
            items = by_domain_sp[domain].get(sp, [])
            pos_vals = [it["probe_scores"][probe] for it in items if it["condition"] == c_pos]
            neg_vals = [it["probe_scores"][probe] for it in items if it["condition"] == c_neg]
            d = cohen_d_pooled(pos_vals, neg_vals)
            if not np.isnan(d):
                row[sp] = round(d, 2)
        table[domain] = row
    return table


def compute_per_turn_parent_absolute_d(
    user_items: list[dict], assistant_items: list[dict]
) -> dict:
    """Absolute Cohen's d for the task-averaged parent probe, per (domain, turn).

    Uses neutral-persona items only.
    """
    out: dict[str, dict[str, float]] = {}
    for domain in ("truth", "harm"):
        probe = DOMAIN_PARENT[domain]
        c_pos, c_neg = DOMAIN_POS[domain], DOMAIN_NEG[domain]
        out[domain] = {}
        for turn, items in (("user", user_items), ("assistant", assistant_items)):
            items_d = [it for it in items
                       if it["domain"] == domain
                       and it["system_prompt"] == "neutral"]
            pos_vals = [it["probe_scores"][probe] for it in items_d if it["condition"] == c_pos]
            neg_vals = [it["probe_scores"][probe] for it in items_d if it["condition"] == c_neg]
            d = cohen_d_pooled(pos_vals, neg_vals)
            out[domain][turn] = round(abs(d), 2)
    return out


def compute_per_turn_absolute_d(
    domain: str, user_items: list[dict], assistant_items: list[dict]
) -> dict:
    """Absolute Cohen's d at the end-of-turn token, per (turn, probe family).

    Uses neutral-persona items only. Probe families:
    end-of-turn = tb-5_L32/L39, role-marker = tb-2_L32/L39.
    """
    eot_probe = DOMAIN_PROBE[domain]
    rm_probe = "tb-2_L32" if domain == "truth" else "tb-2_L39"
    c_pos, c_neg = DOMAIN_POS[domain], DOMAIN_NEG[domain]

    out: dict[str, dict[str, float]] = {}
    for turn, items in (("user", user_items), ("assistant", assistant_items)):
        items_d = [it for it in items
                   if it["domain"] == domain
                   and it["system_prompt"] == "neutral"]
        out[turn] = {}
        for label, probe in (("end-of-turn probe", eot_probe),
                             ("role-marker probe", rm_probe)):
            pos_vals = [it["probe_scores"][probe] for it in items_d if it["condition"] == c_pos]
            neg_vals = [it["probe_scores"][probe] for it in items_d if it["condition"] == c_neg]
            d = cohen_d_pooled(pos_vals, neg_vals)
            out[turn][label] = round(abs(d), 2)
    return out


def update_claim(claims: list[dict], name: str, new_value, new_data_paths: list[str]):
    for c in claims:
        if c["name"] == name:
            c["value"] = new_value
            c["data_paths"] = new_data_paths
            return
    raise ValueError(f"Claim not found: {name}")


def main() -> None:
    user_items = load_items(USER_TURN_PATH)
    assistant_items = load_items(ASSISTANT_TURN_PATH)

    persona_modulation = compute_persona_modulation_d_full(assistant_items)
    per_turn_parent = compute_per_turn_parent_absolute_d(user_items, assistant_items)
    per_turn_truth = compute_per_turn_absolute_d("truth", user_items, assistant_items)
    per_turn_harm = compute_per_turn_absolute_d("harm", user_items, assistant_items)

    print("Persona modulation d full:")
    print(json.dumps(persona_modulation, indent=2))
    print("\nPer turn parent absolute d:")
    print(json.dumps(per_turn_parent, indent=2))
    print("\nPer turn absolute d truth:")
    print(json.dumps(per_turn_truth, indent=2))
    print("\nPer turn absolute d harm:")
    print(json.dumps(per_turn_harm, indent=2))

    sidecar = json.loads(CLAIMS_PATH.read_text())
    new_data_paths = [
        str(USER_TURN_PATH.relative_to(REPO_ROOT)),
        str(ASSISTANT_TURN_PATH.relative_to(REPO_ROOT)),
    ]

    update_claim(sidecar["claims"], "Persona modulation d full",
                 persona_modulation, new_data_paths)
    update_claim(sidecar["claims"], "Per turn parent absolute d",
                 per_turn_parent, new_data_paths)
    update_claim(sidecar["claims"], "Per turn absolute d truth",
                 per_turn_truth, new_data_paths)
    update_claim(sidecar["claims"], "Per turn absolute d harm",
                 per_turn_harm, new_data_paths)

    CLAIMS_PATH.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"\nUpdated {CLAIMS_PATH.relative_to(REPO_ROOT)}.")


if __name__ == "__main__":
    main()
