# NeurIPS 2026 Submission Plan

**Abstract deadline:** 2026-05-04 AOE · **Full paper:** 2026-05-06 AOE · **Track:** Main

## Submission checklist (handbook-derived, 2026-05-06)

Pulled from the NeurIPS 2026 Main Track Handbook + the official `checklist.tex` from `Formatting_Instructions_For_NeurIPS_2026.zip`. Only items the *handbook or submission form* requires; per-author OpenReview profile setup is each author's responsibility and not tracked here.

### Final-pass anonymization checks (do at end, just before submission)

- **Citation hallucination check**

## Infra

- **Sanitised repo for anonymous review release.** Fresh repo (no shared git history with the working repo), containing only what's needed to reproduce paper claims:
  - `src/` — probes, steering, measurement, fitting (drop `experiments/old_experiments/`, weekly reports, reflections, plugin dev, MATS-specific docs)
  - Minimal `experiments/` — only the dirs cited in the paper, with their specs/reports/assets
  - `paper/` — main.tex + claim sidecars + figures (the canonical numbers source)
  - Reproduction instructions: env setup (`uv pip install -e .`), data-prep commands, training script per claim
  - Strip: `.env`, RunPod credentials, internal Slack/voice transcripts, `docs/self_docs/`, `reflections/`, `~/Dev` paths in scripts
  - Anonymise: scrub author names, MATS references, OpenRouter API patterns, repo URL in code comments
  - License: MIT or Apache-2.0; license each new asset (probes, Thurstonian utility data per model)
  - Submission form: anonymous GitHub URL or zip in the supplemental

## Small fixes

- **Unify font sizes, especially in figures.** Tick labels, axis labels, legends, and in-figure annotations vary across panels (matplotlib defaults vs hand-set sizes vs Google-Drawings exports). Pick one set of sizes per role and apply consistently across all figures.
- **Make hero fig "prefill" instead of "say paris".**
- **Clean App.~~\ref{app:shared-openended} to match the new Fig.~~\ref{fig:qualitative-personas}.** The new main-body fig shows four personas (Assistant, Evil, Mathematician, Contrarian) and its caption promises "full transcripts in App.~\ref{app:shared-openended}", but the appendix today only has transcripts for the evil persona at one prompt. Either pull real transcripts for all four personas from `experiments/cross_persona_open_ended_steering/judged_{default,sadist,mathematician,contrarian}.jsonl` (prompts in figure: household-chemicals, photosynthesis, most-interesting-thing, difficult-problems) at $c=0$ and $c=+0.03$ (default at $c=+0.05$) — or trim the main-fig caption claim and keep the appendix as-is.

## Upon acceptance

- **Redraw Fig 3 schematic in matplotlib.** Left third of `fig:steering` is a rasterized SVG (`paper/figures/panels/panels_svg/steering_integrated_mockup.svg`, cropped x=0..365) using Inter at sizes 13/15/17 — different font family AND sizes from the matplotlib panels (DejaVu, 9/10/11/13). Replace `imshow(SCHEMATIC_PNG)` in `paper/figures/panels/build_steering_integrated.py` with a `draw_schematic(ax_sch)` function (Rectangle + ax.text + small badge boxes for ±c). ~30 lines. Eliminates cairosvg rasterization and unifies Fig 3's font world.

## Reference

- `docs/lw_post/lw_post_rendered.md` — methods + results source of truth
- `docs/poster/` — current framing for representation re-use
- `experiments/` — per-experiment figures to pull from

## Followups

(none active)