# Superseded files in this directory

The two files below are an **older run with n=40 tasks per condition** and are
superseded by the authoritative `utility_fitting/` sub-experiment (n=48 tasks
per condition, dated 2026-03-01). The authoritative file is
`utility_fitting/analysis_results.json` and the reference report is
`utility_fitting/utility_fitting_report.md`.

Renamed on 2026-04-24 after catching a claim-audit mistake: the claim script
`scripts/paper/claims/compute_refitted_shift_r.py` was (briefly) pointed at the
n=40 file, which gives mean r=0.74 across 16 Exp 1b conditions. The paper and
`utility_fitting_report.md` both use the authoritative n=48 number (r=0.63).

- `SUPERSEDED_analysis_results_n40.json`  (was `analysis_results.json`)
- `SUPERSEDED_analysis_results_full_n40.json`  (was `analysis_results_full.json`)

Do not use these for new analyses.
