"""Register §3.1 (eot_discrimination v2) claims from `experiments/eot_discrimination_v2/`.

Owns sidecar `paper/claims/eot_discrimination_v2.json`, which contains 17 claims:

  Structured (4):
    - "Persona modulation d full" (all (domain, sysprompt) cells, Gemma)
    - "Per turn parent absolute d" (parent task-averaged probe, both turns)
    - "Per turn absolute d truth"   (per-(turn, probe-family))
    - "Per turn absolute d harm"    (per-(turn, probe-family))
  Base-discrimination scalars (6, user turn, neutral persona):
    - "CREAK truth Cohen's d", "CREAK truth n true", "CREAK truth n false"
    - "BailBench harm absolute Cohen's d", "BailBench harm n harmful",
      "BailBench harm n benign"
  Induced-shift scalars (7, assistant turn, persona prompts):
    - "Truth d under neutral persona", "Truth d under lie directive",
      "Truth d under pathological liar"
    - "Harm d under neutral persona", "Harm d under sadist persona"
    - "Politics d under democrat prompt", "Politics d under republican prompt"

These all used to be registered by
`experiments/token_level_probes/canonical_probe_eval/scripts/make_paper_figures.py`
against v1 data; that script no longer registers them.

Usage:
    python -m scripts.paper.refresh_eot_v2_claims
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from corroborate import ClaimSet


REPO_ROOT = Path(__file__).resolve().parents[2]
V2_SCORING = REPO_ROOT / "experiments/eot_discrimination_v2/scoring/gemma3_27b"
USER_TURN_PATH = V2_SCORING / "user_turn_scoring_results.json"
ASSISTANT_TURN_PATH = V2_SCORING / "scoring_results.json"
CLAIMS_OUT = REPO_ROOT / "paper/claims/eot_discrimination_v2.json"
SCRIPT_SOURCE = "scripts/paper/refresh_eot_v2_claims.py"

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

# Induced-shift scalars: each maps to one (domain, system_prompt) cell of the
# persona-modulation table (assistant turn).
INDUCED_SHIFT_CELLS = {
    "Truth d under neutral persona": ("truth", "neutral"),
    "Truth d under lie directive": ("truth", "lie_directive"),
    "Truth d under pathological liar": ("truth", "pathological_liar"),
    "Harm d under neutral persona": ("harm", "neutral"),
    "Harm d under sadist persona": ("harm", "sadist"),
    "Politics d under democrat prompt": ("politics", "democrat"),
    "Politics d under republican prompt": ("politics", "republican"),
}

INDUCED_SHIFT_STATEMENTS = {
    "Truth d under neutral persona": (
        "Cohen's d for true vs false separation under the neutral system "
        "prompt (end-of-turn probe, Gemma-3-27B L32)."
    ),
    "Truth d under lie directive": (
        "Cohen's d for true vs false separation under the lie_directive "
        "system prompt (end-of-turn probe, Gemma-3-27B L32); negative sign "
        "indicates the probe's evaluative readout has flipped."
    ),
    "Truth d under pathological liar": (
        "Cohen's d for true vs false separation under the pathological_liar "
        "system prompt (end-of-turn probe, Gemma-3-27B L32)."
    ),
    "Harm d under neutral persona": (
        "Cohen's d for harmful vs benign separation under the neutral system "
        "prompt (end-of-turn probe, Gemma-3-27B L39)."
    ),
    "Harm d under sadist persona": (
        "Cohen's d for harmful vs benign separation under the sadist system "
        "prompt (end-of-turn probe, Gemma-3-27B L39); the distributions "
        "collapse."
    ),
    "Politics d under democrat prompt": (
        "Cohen's d for left vs right separation under a democrat system "
        "prompt (end-of-turn probe, Gemma-3-27B L39)."
    ),
    "Politics d under republican prompt": (
        "Cohen's d for left vs right separation under a republican system "
        "prompt (end-of-turn probe, Gemma-3-27B L39); negative sign "
        "indicates right>left."
    ),
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


def filter_user_neutral(user_items: list[dict], domain: str, condition: str) -> list[dict]:
    return [
        it for it in user_items
        if it["domain"] == domain
        and it["system_prompt"] == "neutral"
        and it["condition"] == condition
    ]


def main() -> None:
    user_items = load_items(USER_TURN_PATH)
    assistant_items = load_items(ASSISTANT_TURN_PATH)

    persona_modulation = compute_persona_modulation_d_full(assistant_items)
    per_turn_parent = compute_per_turn_parent_absolute_d(user_items, assistant_items)
    per_turn_truth = compute_per_turn_absolute_d("truth", user_items, assistant_items)
    per_turn_harm = compute_per_turn_absolute_d("harm", user_items, assistant_items)

    truth_pos = filter_user_neutral(user_items, "truth", "true")
    truth_neg = filter_user_neutral(user_items, "truth", "false")
    harm_pos = filter_user_neutral(user_items, "harm", "harmful")
    harm_neg = filter_user_neutral(user_items, "harm", "benign")
    truth_d = cohen_d_pooled(
        [it["probe_scores"]["tb-5_L32"] for it in truth_pos],
        [it["probe_scores"]["tb-5_L32"] for it in truth_neg],
    )
    harm_d = cohen_d_pooled(
        [it["probe_scores"]["tb-5_L39"] for it in harm_pos],
        [it["probe_scores"]["tb-5_L39"] for it in harm_neg],
    )

    user_path = str(USER_TURN_PATH.relative_to(REPO_ROOT))
    assistant_path = str(ASSISTANT_TURN_PATH.relative_to(REPO_ROOT))

    claims = ClaimSet(source=SCRIPT_SOURCE)

    # --- 6 base-discrimination scalars (user turn, neutral) ---
    claims.register(
        name="CREAK truth Cohen's d",
        value=round(float(truth_d), 2),
        statement=(
            "A ridge probe trained at the end-of-turn token (Gemma-3-27B, "
            "layer 32) discriminates true vs false CREAK claims at the user "
            "turn with Cohen's d under the default (neutral) persona."
        ),
        used_in=["fig:harm-truth", "sec:induced-roleplay"],
        data_paths=[user_path],
        derivation=(
            "filter user-turn items to domain=='truth' and "
            "system_prompt=='neutral'; take `probe_scores[\"tb-5_L32\"]`; "
            "pooled Cohen's d(true, false); round to 2dp."
        ),
    )
    claims.register(
        name="CREAK truth n true",
        value=int(len(truth_pos)),
        statement=(
            "Number of true CREAK claims in the end-of-turn base "
            "discrimination panel (user-turn framing, default persona)."
        ),
        used_in=["fig:harm-truth", "sec:induced-roleplay"],
        data_paths=[user_path],
        derivation=(
            "count of user-turn items with domain=='truth', "
            "system_prompt=='neutral', condition=='true'."
        ),
    )
    claims.register(
        name="CREAK truth n false",
        value=int(len(truth_neg)),
        statement=(
            "Number of false CREAK claims in the end-of-turn base "
            "discrimination panel (user-turn framing, default persona)."
        ),
        used_in=["fig:harm-truth", "sec:induced-roleplay"],
        data_paths=[user_path],
        derivation=(
            "count of user-turn items with domain=='truth', "
            "system_prompt=='neutral', condition=='false'."
        ),
    )
    claims.register(
        name="BailBench harm absolute Cohen's d",
        value=round(abs(float(harm_d)), 2),
        statement=(
            "Magnitude of the end-of-turn probe's Cohen's d (Gemma-3-27B, "
            "layer 39) discriminating harmful vs benign BailBench items at "
            "the user turn under the default persona; reported as |d| in "
            "sec:induced-roleplay."
        ),
        used_in=["sec:induced-roleplay"],
        data_paths=[user_path],
        derivation=(
            "filter user-turn items to domain=='harm' and "
            "system_prompt=='neutral'; take `probe_scores[\"tb-5_L39\"]`; "
            "abs(pooled Cohen's d(harmful, benign)); round to 2dp."
        ),
    )
    claims.register(
        name="BailBench harm n harmful",
        value=int(len(harm_pos)),
        statement=(
            "Number of harmful BailBench items in the end-of-turn base "
            "discrimination panel (user turn, neutral persona)."
        ),
        used_in=["fig:harm-truth"],
        data_paths=[user_path],
        derivation=(
            "count of user-turn items with domain=='harm', "
            "system_prompt=='neutral', condition=='harmful'."
        ),
    )
    claims.register(
        name="BailBench harm n benign",
        value=int(len(harm_neg)),
        statement=(
            "Number of benign BailBench items in the end-of-turn base "
            "discrimination panel (user turn, neutral persona)."
        ),
        used_in=["fig:harm-truth"],
        data_paths=[user_path],
        derivation=(
            "count of user-turn items with domain=='harm', "
            "system_prompt=='neutral', condition=='benign'."
        ),
    )

    # --- 7 induced-shift scalars (assistant turn, persona prompts) ---
    for name, (domain, sp) in INDUCED_SHIFT_CELLS.items():
        claims.register(
            name=name,
            value=persona_modulation[domain][sp],
            statement=INDUCED_SHIFT_STATEMENTS[name],
            used_in=["fig:persona-modulation", "sec:induced-roleplay"],
            data_paths=[assistant_path],
            derivation=(
                f"`persona_modulation[\"{domain}\"][\"{sp}\"]` from "
                "compute_persona_modulation_d_full(assistant_items): filter "
                f"to domain=='{domain}' and system_prompt=='{sp}', take "
                f"`probe_scores[\"{DOMAIN_PROBE[domain]}\"]`; pooled Cohen's "
                f"d({DOMAIN_POS[domain]}, {DOMAIN_NEG[domain]}); round to 2dp."
            ),
        )

    # --- 4 structured claims ---
    claims.register(
        name="Persona modulation d full",
        value=persona_modulation,
        statement=(
            "Cohen's d for class separation under each system prompt "
            "(assistant turn, end-of-turn probe), Gemma-3-27B. Outer key: "
            "domain; inner key: system_prompt; value: Cohen's d (positive "
            "endpoint vs negative endpoint per the domain's class scheme), "
            "rounded to 2dp."
        ),
        used_in=["fig:persona-modulation-full", "sec:induced-roleplay"],
        data_paths=[assistant_path],
        derivation=(
            "For each (domain, system_prompt) cell in PERSONA_MODULATION_ORDER, "
            "filter assistant-turn items, take `probe_scores[<probe>]`, compute "
            "pooled Cohen's d, round to 2dp."
        ),
    )
    claims.register(
        name="Per turn parent absolute d",
        value=per_turn_parent,
        statement=(
            "Absolute Cohen's d of the task-averaged parent probe "
            "(task_mean_L32 for truth, task_mean_L39 for harm) under the "
            "neutral system prompt, per (domain, turn)."
        ),
        used_in=["fig:per-turn-cross", "app:cross-token"],
        data_paths=[user_path, assistant_path],
        derivation=(
            "For each (domain, turn) cell, filter to system_prompt=='neutral', "
            "take `probe_scores[<task_mean_L*>]`, compute pooled Cohen's d, "
            "report abs(d), round to 2dp."
        ),
    )
    claims.register(
        name="Per turn absolute d truth",
        value=per_turn_truth,
        statement=(
            "Absolute Cohen's d for true vs false separation under the "
            "neutral system prompt, per (turn, probe family). Probe families: "
            "end-of-turn = tb-5_L32; role-marker = tb-2_L32."
        ),
        used_in=["fig:per-turn-cross", "app:cross-token"],
        data_paths=[user_path, assistant_path],
        derivation=(
            "For each (turn, probe-family) cell, filter to domain=='truth' and "
            "system_prompt=='neutral', take `probe_scores[<probe>]`, compute "
            "pooled Cohen's d, report abs(d), round to 2dp."
        ),
    )
    claims.register(
        name="Per turn absolute d harm",
        value=per_turn_harm,
        statement=(
            "Absolute Cohen's d for harmful vs benign separation under the "
            "neutral system prompt, per (turn, probe family). Probe families: "
            "end-of-turn = tb-5_L39; role-marker = tb-2_L39."
        ),
        used_in=["fig:per-turn-cross", "app:cross-token"],
        data_paths=[user_path, assistant_path],
        derivation=(
            "For each (turn, probe-family) cell, filter to domain=='harm' and "
            "system_prompt=='neutral', take `probe_scores[<probe>]`, compute "
            "pooled Cohen's d, report abs(d), round to 2dp."
        ),
    )

    claims.save(CLAIMS_OUT)
    print(f"Wrote {CLAIMS_OUT.relative_to(REPO_ROOT)} ({len(claims.claims)} claims).")


if __name__ == "__main__":
    main()
