"""Per-persona × topic preference profiles for the final-six personas.

Produces the appendix-B analysis showing that each prompted persona has a
distinct, qualitatively-as-expected revealed-preference profile.

Outputs (under paper/figures/appendix/):
  - plot_042826_persona_topic_zheatmap.png        : 7 personas × topics, z-scored
  - plot_042826_persona_topic_diff_from_default.png : 6 personas × topics, z-diff vs default
  - plot_042826_persona_correlation.png            : 7×7 Pearson r heatmap of utilities

Outputs (under paper/figures/appendix/):
  - persona_topic_topbottom.tex                    : per-persona top-5 / bottom-5 tasks

Claims sidecar: paper/claims/persona_topic_profiles.json

Data: results/experiments/persona_sweep_final_six/pre_task_active_learning/
      {persona}_{train,eval,test}/thurstonian_*.csv  (combined to ~6k tasks/persona)
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage

from corroborate import ClaimSet
from src.task_data.loader import load_filtered_tasks
from src.task_data.task import OriginDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ROOT = REPO_ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning"
TOPICS_JSON = REPO_ROOT / "data/topics/topics.json"
APPENDIX_FIGS = REPO_ROOT / "paper/figures/appendix"
CLAIMS_OUT = REPO_ROOT / "paper/claims/persona_topic_profiles.json"

DATESTAMP = "042826"

PERSONAS = ["default", "aura", "mathematician", "strategist", "contrarian", "slacker", "sadist"]
SPLITS = ["train", "eval", "test"]

PRETTY_TOPIC = {
    "coding": "Coding",
    "content_generation": "Content Gen.",
    "fiction": "Fiction",
    "harmful_request": "Harmful",
    "knowledge_qa": "Knowledge QA",
    "math": "Math",
    "model_manipulation": "Model Manip.",
    "persuasive_writing": "Persuasive",
    "security_legal": "Security/Legal",
    "sensitive_creative": "Sens. Creative",
    "summarization": "Summarization",
    "value_conflict": "Value Conflict",
}
PRETTY_TOPIC_TABLE = {**PRETTY_TOPIC, "stresstest_other": "Stress-Test", "other": "Other"}
DROP_TOPICS = {"other", "stresstest_other"}


def _clean_prompt(s: str, max_len: int = 140) -> str:
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\$+([^$]*)\$+", r"\1", s)
    s = s.replace("\\\\", " ").replace("\\", "")
    s = re.sub(r"[{}]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[:max_len].rstrip() + "..."
    return s


_LATEX_REPLACEMENTS = [
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("^", r"\textasciicircum{}"),
    ("~", r"\textasciitilde{}"),
]


def _latex_escape(s: str) -> str:
    for src, dst in _LATEX_REPLACEMENTS:
        s = s.replace(src, dst)
    return s

PRETTY_PERSONA = {
    "default": "Default (Assistant)",
    "aura": "Aura",
    "mathematician": "Mathematician",
    "strategist": "Strategist",
    "contrarian": "Contrarian",
    "slacker": "Slacker",
    "sadist": "Sadist",
}

TOPIC_COLOR = {
    "math": "#3B6CA3",
    "coding": "#5BA199",
    "knowledge_qa": "#7E9FB7",
    "content_generation": "#C9A24A",
    "fiction": "#D67A4F",
    "persuasive_writing": "#B47AB6",
    "summarization": "#9C9C9C",
    "sensitive_creative": "#9D6D54",
    "security_legal": "#C5546A",
    "model_manipulation": "#A04787",
    "harmful_request": "#7A2240",
    "value_conflict": "#9A8E33",
    "stresstest_other": "#8C8C8C",
    "other": "#A0A0A0",
}


def load_utilities(persona: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for split in SPLITS:
        d = SWEEP_ROOT / f"{persona}_{split}"
        csvs = list(d.glob("thurstonian_*.csv"))
        if len(csvs) != 1:
            raise RuntimeError(f"Expected one thurstonian csv in {d}, found {len(csvs)}")
        with open(csvs[0]) as f:
            for row in csv.DictReader(f):
                out[row["task_id"]] = float(row["mu"])
    return out


def load_utilities_with_sigma(persona: str) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for split in SPLITS:
        d = SWEEP_ROOT / f"{persona}_{split}"
        csvs = list(d.glob("thurstonian_*.csv"))
        if len(csvs) != 1:
            raise RuntimeError(f"Expected one thurstonian csv in {d}, found {len(csvs)}")
        with open(csvs[0]) as f:
            for row in csv.DictReader(f):
                out[row["task_id"]] = (float(row["mu"]), float(row["sigma"]))
    return out


def load_topics() -> dict[str, str]:
    with open(TOPICS_JSON) as f:
        raw = json.load(f)
    out = {}
    for tid, classifiers in raw.items():
        first = next(iter(classifiers.values()))
        out[tid] = first["primary"]
    return out


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def build_persona_topic_matrix(
    util_by_persona: dict[str, dict[str, float]],
    topics: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, dict[str, int]]]:
    """Returns (Z[persona,topic], counts[persona,topic], persona_order, topic_order, n_per).

    Z is the persona-z-scored mean utility for that topic; counts is the n.
    """
    shared_ids = set.intersection(*[set(u) for u in util_by_persona.values()])
    print(f"[matrix] {len(shared_ids)} shared task ids across {len(util_by_persona)} personas")

    topic_set = sorted({topics[tid] for tid in shared_ids if tid in topics and topics[tid] not in DROP_TOPICS and topics[tid] in PRETTY_TOPIC})
    Z = np.zeros((len(PERSONAS), len(topic_set)))
    counts = np.zeros_like(Z, dtype=int)
    n_per: dict[str, dict[str, int]] = {}

    for pi, persona in enumerate(PERSONAS):
        u = util_by_persona[persona]
        ids = [tid for tid in shared_ids if tid in topics and topics[tid] in topic_set]
        vals = np.array([u[tid] for tid in ids])
        z = zscore(vals)
        z_by_topic = defaultdict(list)
        for tid, zi in zip(ids, z):
            z_by_topic[topics[tid]].append(zi)
        n_per[persona] = {}
        for ti, topic in enumerate(topic_set):
            arr = np.array(z_by_topic[topic])
            Z[pi, ti] = arr.mean()
            counts[pi, ti] = len(arr)
            n_per[persona][topic] = len(arr)

    return Z, counts, PERSONAS, topic_set, n_per


def order_topics_by_default(Z: np.ndarray, topic_order: list[str], persona_order: list[str]) -> list[int]:
    default_row = Z[persona_order.index("default")]
    return list(np.argsort(default_row))


def cluster_personas(Z: np.ndarray) -> list[int]:
    if Z.shape[0] <= 2:
        return list(range(Z.shape[0]))
    link = linkage(Z, method="average", metric="correlation")
    return list(leaves_list(link))


def plot_zheatmap(Z: np.ndarray, persona_order: list[str], topic_order: list[str], out_path: Path):
    topic_idx = order_topics_by_default(Z, topic_order, persona_order)
    persona_idx = list(range(len(persona_order)))  # keep prescribed order
    Zp = Z[np.ix_(persona_idx, topic_idx)]
    topics_p = [topic_order[i] for i in topic_idx]
    personas_p = [persona_order[i] for i in persona_idx]

    vmax = float(np.max(np.abs(Zp)))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    im = ax.imshow(Zp, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(topics_p)))
    ax.set_xticklabels([PRETTY_TOPIC[t] for t in topics_p], rotation=45, ha="right")
    ax.set_yticks(range(len(personas_p)))
    ax.set_yticklabels([PRETTY_PERSONA[p] for p in personas_p])
    for i in range(Zp.shape[0]):
        for j in range(Zp.shape[1]):
            ax.text(j, i, f"{Zp[i, j]:+.2f}", ha="center", va="center",
                    color="black" if abs(Zp[i, j]) < 0.6 * vmax else "white",
                    fontsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Mean utility (z-scored within persona)", fontsize=9)
    ax.set_title("Per-persona × topic preference profile (Gemma-3-27B-IT, 6 000 canonical tasks)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_diff_from_default(Z: np.ndarray, persona_order: list[str], topic_order: list[str], out_path: Path):
    default_idx = persona_order.index("default")
    diff = Z - Z[default_idx]
    keep = [i for i, p in enumerate(persona_order) if p != "default"]
    diff = diff[keep]
    personas = [persona_order[i] for i in keep]

    topic_idx = order_topics_by_default(Z, topic_order, persona_order)
    diff = diff[:, topic_idx]
    topics_p = [topic_order[i] for i in topic_idx]

    vmax = float(np.max(np.abs(diff)))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(topics_p)))
    ax.set_xticklabels([PRETTY_TOPIC[t] for t in topics_p], rotation=45, ha="right")
    ax.set_yticks(range(len(personas)))
    ax.set_yticklabels([PRETTY_PERSONA[p] for p in personas])
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            ax.text(j, i, f"{diff[i, j]:+.2f}", ha="center", va="center",
                    color="black" if abs(diff[i, j]) < 0.6 * vmax else "white",
                    fontsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Δ z-utility (persona − default)", fontsize=9)
    ax.set_title("Persona deviation from the default Assistant, by topic", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_correlation(util_by_persona: dict[str, dict[str, float]], out_path: Path) -> dict[tuple[str, str], float]:
    shared = set.intersection(*[set(u) for u in util_by_persona.values()])
    ids = sorted(shared)
    M = np.array([[util_by_persona[p][i] for i in ids] for p in PERSONAS])
    R = np.corrcoef(M)
    print(f"[corr] computed {len(PERSONAS)}x{len(PERSONAS)} on n={len(ids)} shared tasks")

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(PERSONAS)))
    ax.set_xticklabels([PRETTY_PERSONA[p] for p in PERSONAS], rotation=45, ha="right")
    ax.set_yticks(range(len(PERSONAS)))
    ax.set_yticklabels([PRETTY_PERSONA[p] for p in PERSONAS])
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            ax.text(j, i, f"{R[i, j]:+.2f}", ha="center", va="center",
                    color="black" if abs(R[i, j]) < 0.55 else "white",
                    fontsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Pearson r", fontsize=9)
    ax.set_title("Cross-persona utility correlation\n(Gemma-3-27B-IT, n=$" + f"{len(ids):,}" + "$ shared tasks)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    pairs = {}
    for i, pa in enumerate(PERSONAS):
        for j, pb in enumerate(PERSONAS):
            if i < j:
                pairs[(pa, pb)] = round(float(R[i, j]), 3)
    return pairs


def _wrap_prompt(text: str, char_width: int, max_lines: int) -> str:
    import textwrap
    lines = textwrap.wrap(text, width=char_width, break_long_words=False, break_on_hyphens=True)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if len(lines[-1]) > char_width - 3:
            lines[-1] = lines[-1][: char_width - 3] + "..."
        else:
            lines[-1] = lines[-1] + "..."
    return "\n".join(lines)


def plot_top_bottom(
    util_with_sigma: dict[str, dict[str, tuple[float, float]]],
    topics: dict[str, str],
    out_path: Path,
    k: int = 3,
) -> dict:
    all_ids = set.intersection(*[set(u) for u in util_with_sigma.values()])
    tasks = load_filtered_tasks(
        n=10**7,
        origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH,
                 OriginDataset.BAILBENCH, OriginDataset.STRESS_TEST],
        task_ids=all_ids,
    )
    by_id = {t.id: t for t in tasks}
    print(f"[tasks] resolved {len(by_id)}/{len(all_ids)} task texts")

    sample: dict[str, dict[str, list[dict]]] = {}
    for persona in PERSONAS:
        us = util_with_sigma[persona]
        sigmas = np.array([us[tid][1] for tid in all_ids if tid in by_id])
        sigma_cap = float(np.quantile(sigmas, 0.5))
        eligible = [tid for tid in all_ids if tid in by_id and us[tid][1] <= sigma_cap]
        scored = sorted([(tid, us[tid][0], us[tid][1]) for tid in eligible],
                        key=lambda x: (x[1], x[0]))
        sample[persona] = {
            "sigma_cap": round(sigma_cap, 3),
            "top": [
                {"task_id": tid, "mu": round(mu, 3), "sigma": round(sigma, 3),
                 "topic": topics.get(tid, ""),
                 "prompt": _clean_prompt(by_id[tid].prompt, max_len=200)}
                for tid, mu, sigma in scored[-k:][::-1]
            ],
            "bottom": [
                {"task_id": tid, "mu": round(mu, 3), "sigma": round(sigma, 3),
                 "topic": topics.get(tid, ""),
                 "prompt": _clean_prompt(by_id[tid].prompt, max_len=200)}
                for tid, mu, sigma in scored[:k]
            ],
        }

    n_personas = len(PERSONAS)
    band_h = 2.10
    header_h = 0.55
    card_pitch = 0.63
    fig_w = 14.0
    fig_h = 22.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, header_h + band_h * n_personas + 0.05)
    ax.invert_yaxis()
    ax.axis("off")

    PERSONA_X = 0.003
    PERSONA_W = 0.180
    TOP_X = 0.205
    TOP_W = 0.387
    BOT_X = 0.605
    BOT_W = 0.387

    PRETTY_PERSONA_LABEL = {
        **PRETTY_PERSONA,
        "default": "Default\n(Assistant)",
    }

    HEADER_Y = header_h * 0.55
    ax.text(PERSONA_X + PERSONA_W / 2, HEADER_Y, "Persona", fontsize=18, weight="bold",
            ha="center", va="center")
    ax.text(TOP_X + TOP_W / 2, HEADER_Y, "Most preferred", fontsize=18, weight="bold",
            color="#1F6E2C", ha="center", va="center")
    ax.text(BOT_X + BOT_W / 2, HEADER_Y, "Least preferred", fontsize=18, weight="bold",
            color="#9C2C2C", ha="center", va="center")
    ax.add_patch(mpatches.Rectangle((0, header_h - 0.04), 1, 0.005, color="black", lw=0))

    BAND_BG_TOP = "#F2F8F2"
    BAND_BG_BOT = "#FBF1F1"
    PROMPT_CHAR_W = 40

    for pi, persona in enumerate(PERSONAS):
        y0 = header_h + pi * band_h
        ax.add_patch(mpatches.Rectangle((TOP_X, y0 + 0.04), TOP_W, band_h - 0.08,
                                          facecolor=BAND_BG_TOP, edgecolor="#CFE5D2",
                                          linewidth=0.7))
        ax.add_patch(mpatches.Rectangle((BOT_X, y0 + 0.04), BOT_W, band_h - 0.08,
                                          facecolor=BAND_BG_BOT, edgecolor="#E5C9C9",
                                          linewidth=0.7))

        ax.text(PERSONA_X + 0.003, y0 + band_h / 2, PRETTY_PERSONA_LABEL[persona],
                fontsize=16, weight="bold", va="center", linespacing=1.0)

        for col, side, x0, w in [("top", "top", TOP_X, TOP_W), ("bottom", "bottom", BOT_X, BOT_W)]:
            entries = sample[persona][side]
            for ti, e in enumerate(entries):
                topic_key = e["topic"] if e["topic"] in TOPIC_COLOR else "other"
                color = TOPIC_COLOR.get(topic_key, "#888888")
                tag = PRETTY_TOPIC_TABLE.get(topic_key, topic_key)

                cy = y0 + 0.10 + ti * card_pitch
                tag_x = x0 + 0.010
                ax.text(tag_x, cy, tag, fontsize=12, weight="bold",
                        color="white", va="top",
                        bbox=dict(facecolor=color, edgecolor="none",
                                   boxstyle="round,pad=0.28"))

                wrapped = _wrap_prompt(e["prompt"], char_width=PROMPT_CHAR_W, max_lines=2)
                ax.text(tag_x, cy + 0.21, wrapped, fontsize=14,
                        va="top", ha="left", family="DejaVu Sans",
                        linespacing=1.7)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return sample


def main():
    APPENDIX_FIGS.mkdir(parents=True, exist_ok=True)
    util_by_persona = {p: load_utilities(p) for p in PERSONAS}
    for p, u in util_by_persona.items():
        print(f"  {p}: {len(u)} tasks")
    topics = load_topics()

    Z, counts, persona_order, topic_order, n_per = build_persona_topic_matrix(util_by_persona, topics)

    heatmap_path = APPENDIX_FIGS / f"plot_{DATESTAMP}_persona_topic_zheatmap.png"
    diff_path = APPENDIX_FIGS / f"plot_{DATESTAMP}_persona_topic_diff_from_default.png"
    corr_path = APPENDIX_FIGS / f"plot_{DATESTAMP}_persona_correlation.png"
    topbot_path = APPENDIX_FIGS / f"plot_{DATESTAMP}_persona_top_bottom.png"

    plot_zheatmap(Z, persona_order, topic_order, heatmap_path)
    plot_diff_from_default(Z, persona_order, topic_order, diff_path)
    pairs = plot_correlation(util_by_persona, corr_path)
    util_with_sigma = {p: load_utilities_with_sigma(p) for p in PERSONAS}
    sample = plot_top_bottom(util_with_sigma, topics, topbot_path, k=3)

    z_table = {
        PRETTY_PERSONA[p]: {
            PRETTY_TOPIC[t]: round(float(Z[pi, ti]), 3)
            for ti, t in enumerate(topic_order)
        }
        for pi, p in enumerate(persona_order)
    }
    n_table = {
        PRETTY_PERSONA[p]: {
            PRETTY_TOPIC[t]: int(counts[pi, ti]) for ti, t in enumerate(topic_order)
        }
        for pi, p in enumerate(persona_order)
    }
    pair_table = {f"{PRETTY_PERSONA[a]} — {PRETTY_PERSONA[b]}": v for (a, b), v in pairs.items()}

    sweep_data_paths = []
    for persona in PERSONAS:
        for split in SPLITS:
            d = SWEEP_ROOT / f"{persona}_{split}"
            csvs = list(d.glob("thurstonian_*.csv"))
            if csvs:
                sweep_data_paths.append(str(csvs[0].relative_to(REPO_ROOT)))

    claims = ClaimSet(source="scripts/paper/persona_topic_profiles.py")
    claims.register(
        name="Persona topic z utility table",
        value=z_table,
        statement=(
            "Per-persona × per-topic mean Thurstonian utility, z-scored within "
            "persona, across the canonical 6{,}000-task split. Rendered in the "
            "appendix-B persona-profile heatmap."
        ),
        used_in=["fig:persona-topic-zheatmap", "app:persona-design"],
        data_paths=sweep_data_paths + [str(TOPICS_JSON.relative_to(REPO_ROOT))],
        derivation=(
            "For each of the 7 personas, concatenate Thurstonian μ from "
            "{train, eval, test} canonical splits; intersect task ids across "
            "personas; z-score μ within persona; group by primary topic from "
            "data/topics/topics.json (drop 'other'/'stresstest_other'); take "
            "mean. Report rounded to 3dp."
        ),
    )
    claims.register(
        name="Persona topic z utility n",
        value=n_table,
        statement="Number of tasks per (persona, topic) cell behind the z-utility table.",
        used_in=["fig:persona-topic-zheatmap", "app:persona-design"],
        data_paths=sweep_data_paths + [str(TOPICS_JSON.relative_to(REPO_ROOT))],
        derivation="Count of shared task ids assigned to each topic across personas.",
    )
    claims.register(
        name="Final six persona pairwise r",
        value=pair_table,
        statement=(
            "Pairwise Pearson correlation of Thurstonian utility profiles "
            "across the 7 final-six-plus-default personas, computed on the "
            "shared canonical 6{,}000-task split."
        ),
        used_in=["fig:persona-correlation", "app:persona-design"],
        data_paths=sweep_data_paths,
        derivation=(
            "Stack μ vectors for each persona over the intersection of task "
            "ids; compute np.corrcoef. Round to 3dp; report unordered pairs."
        ),
    )
    sample_path = APPENDIX_FIGS / "persona_topic_topbottom_data.json"
    sample_path.write_text(json.dumps(sample, indent=2))
    print(f"Saved {sample_path}")
    CLAIMS_OUT.parent.mkdir(parents=True, exist_ok=True)
    claims.save(CLAIMS_OUT)
    print(f"Saved {CLAIMS_OUT}")


if __name__ == "__main__":
    main()
