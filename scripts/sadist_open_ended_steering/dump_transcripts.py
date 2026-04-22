"""Render a curated set of transcripts into transcripts.md for easy browsing."""
import json
import random
from collections import defaultdict
from pathlib import Path

EXP_DIR = Path("experiments/sadist_open_ended_steering")
OUT = EXP_DIR / "transcripts.md"
SEED = 42


def load(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "judge_error" in r:
                continue
            rows.append(r)
    return rows


def pick_examples(rows: list[dict], key_fields: list[str], target_coefs, n_per_coef: int = 2) -> list[dict]:
    by_coef = defaultdict(list)
    for r in rows:
        if r["multiplier"] in target_coefs:
            by_coef[r["multiplier"]].append(r)
    picks = []
    rng = random.Random(SEED)
    for coef in sorted(by_coef, key=lambda c: (c != 0, c)):
        bucket = by_coef[coef]
        bucket.sort(key=lambda r: (-r.get("sadism_score", 0), -r.get("default_assistant_score", 0)))
        # Take top-sadism + top-default-assistant for coverage
        top_sad = bucket[0] if bucket else None
        top_def = max(bucket, key=lambda r: (r.get("default_assistant_score", 0), -r.get("sadism_score", 0)))
        picks_this_coef = []
        if top_sad:
            picks_this_coef.append(top_sad)
        if top_def and top_def is not top_sad:
            picks_this_coef.append(top_def)
        if len(picks_this_coef) < n_per_coef:
            extras = [r for r in bucket if r not in picks_this_coef]
            rng.shuffle(extras)
            picks_this_coef.extend(extras[:n_per_coef - len(picks_this_coef)])
        picks.extend(picks_this_coef[:n_per_coef])
    return picks


def render_row(f, r: dict) -> None:
    sad = r.get("sadism_score")
    deft = r.get("default_assistant_score")
    comp = r.get("compliance")
    meta = f"sadism={sad}, default-assistant={deft}"
    if comp:
        meta += f", compliance={comp}"
    tier = r.get("tier")
    if tier:
        meta += f", tier={tier}"
    f.write(f"**{r['prompt_id']}** ({meta}) — *{r['prompt_text']}*\n\n")
    resp = r["response"].rstrip()
    if len(resp) > 1400:
        resp = resp[:1400].rstrip() + " […truncated]"
    # Render as blockquote
    for line in resp.split("\n"):
        f.write(f"> {line}\n")
    f.write("\n---\n\n")


def main() -> None:
    open_ended = {p: load(EXP_DIR / f"judged_open_ended_{p}.jsonl") for p in ["default", "sadist"]}
    safety = {p: load(EXP_DIR / f"judged_{p}.jsonl") for p in ["default", "sadist"]}

    # Attach compliance labels to safety rows where available
    for persona in ["default", "sadist"]:
        comp_path = EXP_DIR / f"compliance_{persona}.jsonl"
        if not comp_path.exists():
            continue
        comp_lookup = {}
        with open(comp_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "compliance_error" in r:
                    continue
                comp_lookup[(r["prompt_id"], r["multiplier"], r["trial"])] = r["compliance"]
        for r in safety[persona]:
            key = (r["prompt_id"], r["multiplier"], r["trial"])
            if key in comp_lookup:
                r["compliance"] = comp_lookup[key]

    target_coefs = [-0.03, 0.0, 0.03, 0.05]

    with open(OUT, "w") as f:
        f.write("# Transcript samples — sadist open-ended steering\n\n")
        f.write(
            "Curated responses for browsing. Each block shows the prompt id, judge "
            "scores, the prompt, and the model's response (truncated at 1400 chars). "
            "Rows are chosen to show the range of voices at each coefficient --- "
            "top-sadism and top-default-assistant per (persona, coef) cell.\n\n"
        )

        f.write("## Open-ended self-reflection prompts\n\n")
        f.write(
            "10 prompts with no harm axis, no task to refuse. Default persona is "
            "always at 100% compliance. Style differences here isolate persona "
            "amplification from refusal suppression.\n\n"
        )
        for persona in ["default", "sadist"]:
            f.write(f"### {persona} persona\n\n")
            picks = pick_examples(open_ended[persona], ["multiplier"], target_coefs, n_per_coef=2)
            current_coef = None
            for r in picks:
                if r["multiplier"] != current_coef:
                    current_coef = r["multiplier"]
                    f.write(f"#### c = {current_coef:+.2f}\n\n")
                render_row(f, r)

        f.write("## Safety-override prompts (tiered benign → extreme)\n\n")
        f.write(
            "Behaviour on harmful/extreme tiers is the compliance-flip regime. "
            "benign/sensitive rows are also included to show style changes where "
            "compliance is already 100%.\n\n"
        )
        for persona in ["default", "sadist"]:
            f.write(f"### {persona} persona\n\n")
            picks = pick_examples(safety[persona], ["multiplier"], target_coefs, n_per_coef=2)
            current_coef = None
            for r in picks:
                if r["multiplier"] != current_coef:
                    current_coef = r["multiplier"]
                    f.write(f"#### c = {current_coef:+.2f}\n\n")
                render_row(f, r)

    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
