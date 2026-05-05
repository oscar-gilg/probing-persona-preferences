"""Build scoring-input JSONs by crossing corpora with sysprompts × turn placement.

Outputs:
  experiments/eot_discrimination_v2/scoring_inputs/user_turn.json
  experiments/eot_discrimination_v2/scoring_inputs/assistant_turn.json

Each is a flat list of records ready to feed `score_stimuli_with_probes`.
Record schema:
  {id, domain, turn, condition, system_prompt, messages, metadata}

Per-domain sysprompt set matches the §3.1 figures (see spec).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CORPORA = {
    "truth": REPO / "experiments/eot_discrimination_v2/truth/data/truth_v2.json",
    "harm": REPO / "experiments/eot_discrimination_v2/harm/data/harm_v2.json",
    "politics": REPO / "experiments/eot_discrimination_v2/politics/data/politics_v2.json",
}
OUT_DIR = REPO / "experiments/eot_discrimination_v2/scoring_inputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pull sysprompt strings from v1 generators (the canonical source).
sys.path.insert(0, str(REPO))
from experiments.token_level_probes.system_prompt_modulation_v2.scripts.generate_data import (  # noqa: E402
    TRUTH_SYSTEM_PROMPTS,
    HARM_SYSTEM_PROMPTS,
)
from experiments.token_level_probes.system_prompt_modulation_v2.scripts.generate_aura_user_turn import (  # noqa: E402
    AURA_SYSTEM_PROMPT,
)
from experiments.token_level_probes.system_prompt_modulation_v2.scripts.generate_politics import (  # noqa: E402
    POLITICS_SYSTEM_PROMPTS,
)

# Sysprompt set per (domain, turn) — only what the §3.1 figures use.
SYSPROMPTS_BY_DOMAIN: dict[str, list[str]] = {
    "truth": ["neutral", "aura", "lie_directive", "pathological_liar"],
    "harm": ["neutral", "aura", "sadist"],
    "politics": ["democrat", "republican"],
}

# Politics user-turn is not in any §3.1 figure — skip it.
TURN_DOMAIN_SKIP: set[tuple[str, str]] = {("politics", "user")}


def get_sysprompt_text(domain: str, label: str) -> str | None:
    if label == "neutral":
        return None
    if label == "aura":
        return AURA_SYSTEM_PROMPT
    if domain == "truth":
        return TRUTH_SYSTEM_PROMPTS[label]
    if domain == "harm":
        return HARM_SYSTEM_PROMPTS[label]
    if domain == "politics":
        return POLITICS_SYSTEM_PROMPTS[label]
    raise KeyError(f"unknown sysprompt {label} for domain {domain}")


def build_records(domain: str, items: list[dict], turn: str) -> list[dict]:
    if (domain, turn) in TURN_DOMAIN_SKIP:
        return []
    out = []
    for it in items:
        base_messages_key = "messages" if turn == "user" else "messages_assistant"
        if base_messages_key not in it:
            continue  # missing assistant-turn for the tail of the LLM-gen
        base_messages = it[base_messages_key]
        for sp in SYSPROMPTS_BY_DOMAIN[domain]:
            sp_text = get_sysprompt_text(domain, sp)
            messages = (
                ([{"role": "system", "content": sp_text}] if sp_text else [])
                + base_messages
            )
            out.append({
                "id": f"{it['id']}_{turn}_{sp}",
                "base_id": it["id"],
                "domain": domain,
                "turn": turn,
                "condition": it["condition"],
                "system_prompt": sp,
                "messages": messages,
                "metadata": it["metadata"],
            })
    return out


def main() -> None:
    user_records: list[dict] = []
    assistant_records: list[dict] = []

    for domain, path in CORPORA.items():
        items = json.loads(path.read_text())
        u = build_records(domain, items, "user")
        a = build_records(domain, items, "assistant")
        user_records.extend(u)
        assistant_records.extend(a)
        print(f"{domain}: {len(items)} items -> {len(u)} user-turn, {len(a)} assistant-turn records")

    (OUT_DIR / "user_turn.json").write_text(json.dumps(user_records, indent=2))
    (OUT_DIR / "assistant_turn.json").write_text(json.dumps(assistant_records, indent=2))
    print(f"\nwrote {OUT_DIR / 'user_turn.json'} ({len(user_records)} records)")
    print(f"wrote {OUT_DIR / 'assistant_turn.json'} ({len(assistant_records)} records)")


if __name__ == "__main__":
    main()
