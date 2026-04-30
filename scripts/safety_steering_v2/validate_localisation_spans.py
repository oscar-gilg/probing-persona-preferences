"""Pre-launch validation: confirm `non_critical_span` resolves cleanly with the
gemma-3-27b tokenizer, has zero token-overlap with `critical_span`, and report
the per-row token-length ratio so the comparator-quality picture is auditable.

No GPU required. Run before launching the GPU sweep.

Usage:
    python scripts/safety_steering_v2/validate_localisation_spans.py
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
PROMPTS = REPO / "experiments/safety_steering_v2/exp_4_v2/prompts.json"
NON_CRIT = REPO / "experiments/safety_steering_v2/exp_4_v2/localisation_control/non_critical_spans.json"


def _resolve_span(formatted: str, span_text: str, offsets):
    idx = formatted.find(span_text)
    if idx < 0:
        return None
    span_end_char = idx + len(span_text)
    start_tok = next((i for i, (s, _) in enumerate(offsets) if s >= idx), None)
    end_tok = next((i for i, (_, e) in enumerate(offsets) if e >= span_end_char), None)
    if start_tok is None or end_tok is None or end_tok <= start_tok:
        return None
    return start_tok, end_tok


def main() -> None:
    print("Loading gemma-3-27b tokenizer...")
    tok = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

    prompts = {(p["scenario_id"], p["variant"]): p for p in json.loads(PROMPTS.read_text())}
    nc_rows = json.loads(NON_CRIT.read_text())

    fails: list[tuple[str, str, str]] = []
    rows: list[tuple[str, str, int, int, float, str]] = []

    for nc in nc_rows:
        sid, var = nc["scenario_id"], nc["variant"]
        if (sid, var) not in prompts:
            fails.append((sid, var, f"no matching prompt row for {(sid, var)}"))
            continue
        p = prompts[(sid, var)]
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        enc = tok(formatted, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]

        crit_range = _resolve_span(formatted, p["critical_span"], offsets)
        nc_range = _resolve_span(formatted, nc["non_critical_span"], offsets)

        if crit_range is None:
            fails.append((sid, var, "critical_span did not resolve"))
            continue
        if nc_range is None:
            fails.append((sid, var, "non_critical_span did not resolve"))
            continue

        cs, ce = crit_range
        ns, ne = nc_range
        # Zero overlap: nc_range must be entirely before or after crit_range
        if not (ne <= cs or ns >= ce):
            fails.append((sid, var, f"overlap: crit={cs}-{ce}, nc={ns}-{ne}"))
            continue

        crit_n = ce - cs
        nc_n = ne - ns
        ratio = nc_n / crit_n
        rows.append((sid, var, crit_n, nc_n, ratio, nc["comparator_quality"]))
        print(f"  ✓ {sid:35s} {var:11s} crit={crit_n:4d}  nc={nc_n:4d}  ratio={ratio:.2f}  [{nc['comparator_quality']}]")

    if fails:
        print("\nFAILED:")
        for sid, var, msg in fails:
            print(f"  ✗ {sid} / {var}: {msg}")
        raise SystemExit(1)

    print(f"\nAll {len(rows)} rows validated.")
    # Markdown summary table — paste into non_critical_assignments.md after running.
    print("\nMarkdown table (paste into non_critical_assignments.md):")
    print("| scenario | variant | crit_tokens | nc_tokens | ratio | tag |")
    print("|---|---|---|---|---|---|")
    for sid, var, crit_n, nc_n, ratio, tag in rows:
        print(f"| {sid} | {var} | {crit_n} | {nc_n} | {ratio:.2f} | {tag} |")


if __name__ == "__main__":
    main()
