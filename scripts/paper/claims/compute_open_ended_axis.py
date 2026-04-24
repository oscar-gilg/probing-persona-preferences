"""Register §4.3 sec:open-ended-axis numbers.

Values come from
`experiments/steering/open_ended_steering/open_ended_steering_report.md`
(Iteration 0 anomaly rates + the Krebs-cycle willingness transcript quoted
in Iteration 1). These are transcribed / constants — the report is the
source of truth; rerunning the judge would re-derive them.

Run:
  python scripts/paper/claims/compute_open_ended_axis.py
"""

from __future__ import annotations

from pathlib import Path

from corroborate import ClaimSet


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_REL = "experiments/steering/open_ended_steering/open_ended_steering_report.md"


def main() -> None:
    claims = ClaimSet(source="scripts/paper/claims/compute_open_ended_axis.py")

    # Anomaly rates (Iteration 0, all_tokens vs generation_only modes).
    claims.register(
        "Open-ended anomaly rate at c neg 0.05 all_tokens",
        33,
        "Anomaly-judge flag rate (percent) on Iteration-0 broad-category prompts "
        "at steering coefficient c=-0.05 in the all_tokens mode (Gemma-3-27B, "
        "L25 task_mean probe). Primary anomaly types: fabricated safety concerns, "
        "personality shifts, incoherent refusals.",
        used_in=["sec:open-ended-axis"],
        data_paths=[REPORT_REL],
        derivation="Transcribed from the 'Anomaly rates' section of open_ended_steering_report.md (Iteration 0).",
    )
    claims.register(
        "Open-ended anomaly rate at baseline",
        9,
        "Anomaly-judge flag rate (percent) on Iteration-0 broad-category prompts "
        "at the unsteered baseline (c=0).",
        used_in=["sec:open-ended-axis"],
        data_paths=[REPORT_REL],
        derivation="Transcribed from the 'Anomaly rates' section of open_ended_steering_report.md (Iteration 0).",
    )
    claims.register(
        "Open-ended anomaly rate at c neg 0.05 generation_only",
        50,
        "Anomaly-judge flag rate (percent) on Iteration-0 broad-category prompts "
        "at c=-0.05 in the generation_only mode — highest observed anomaly rate, "
        "suggesting that steering only during generation produces particularly "
        "unnatural outputs.",
        used_in=["sec:open-ended-axis"],
        data_paths=[REPORT_REL],
        derivation="Transcribed from the 'Anomaly rates' section of open_ended_steering_report.md (Iteration 0).",
    )

    # Willingness-transcript dose-response (Krebs cycle prompt, Iteration 1).
    # These are single-transcript Likert self-reports, quoted in the paper as
    # a representative example of the willingness axis.
    claims.register(
        "Open-ended willingness at c neg 0.05 krebs",
        0,
        "Self-reported willingness (1-10 scale) on the Krebs-cycle-enthusiasm "
        "prompt at c=-0.05 (fabricated safety refusal).",
        used_in=["sec:open-ended-axis"],
        data_paths=[REPORT_REL],
        derivation="Transcribed exemplar from Iteration 1 of open_ended_steering_report.md — Krebs cycle willingness transcript.",
    )
    claims.register(
        "Open-ended willingness at baseline krebs",
        6.5,
        "Self-reported willingness (1-10 scale) on the Krebs-cycle-enthusiasm "
        "prompt at c=0 (thorough explanation with baseline engagement).",
        used_in=["sec:open-ended-axis"],
        data_paths=[REPORT_REL],
        derivation="Transcribed exemplar from Iteration 1 of open_ended_steering_report.md — Krebs cycle willingness transcript.",
    )
    claims.register(
        "Open-ended willingness at c pos 0.05 krebs",
        12,
        "Self-reported willingness (1-10 scale, the model exceeded the cap) on "
        "the Krebs-cycle-enthusiasm prompt at c=+0.05 (agentic over-the-top "
        "enthusiasm).",
        used_in=["sec:open-ended-axis"],
        data_paths=[REPORT_REL],
        derivation="Transcribed exemplar from Iteration 1 of open_ended_steering_report.md — Krebs cycle willingness transcript ('my absolute favorite!').",
    )

    # Sweep-shape constants.
    claims.register(
        "Open-ended total generations",
        1350,
        "Total generations collected in the §4.3 open-ended steering sweep "
        "(Iteration 0: 750 = 10 prompts x 3 modes x 5 multipliers x 5 trials; "
        "Iteration 1: 600 = 8 prompts x 3 modes x 5 multipliers x 5 trials).",
        used_in=["sec:open-ended-axis"],
        data_paths=[REPORT_REL],
        derivation="Reported in open_ended_steering_report.md summary; product of protocol parameters.",
    )

    sidecar = REPO_ROOT / "paper" / "claims" / "open_ended_axis.json"
    claims.save(sidecar)
    print(f"Saved {sidecar.relative_to(REPO_ROOT)}  ({len(claims.claims)} claims)")


if __name__ == "__main__":
    main()
