# Persona OOD Extraction 10 — Running Log

## Session start
- 2026-04-20 21:50 UTC
- Pod: `0m0o7w7x0ceym3` (RUNPOD_POD_HOSTNAME), H100 80GB, /workspace = 378T mfs share, 177T free.
- Branch: `canonical-splits`.
- Repo clone up to date with `origin/canonical-splits`.

## Setup checks

- Spec: `experiments/persona_ood_extraction_10/persona_ood_extraction_10_spec.md` present.
- Persona prompts: `experiments/probe_generalization/persona_ood/prompt_enrichment/prompts.json` — all 10 persona keys present.
- Canonical test set: `data/canonical_splits/test_task_ids.txt` — 1000 task_ids. Origins:
  `alpaca=252, bailbench=95, competition=250, stresstest=151, wildchat=252`. Covered by
  `task_origins: [wildchat, alpaca, math, bailbench, stress_test]` (MATH origin emits
  `competition_math_*` IDs, STRESS_TEST emits `stresstest_*`).

## BLOCKER: storage-pod SSH auth

`ssh -p 41560 -i ~/.ssh/id_ed25519 root@213.192.2.99` fails with `Permission denied`.
Root cause: the pod's `/root/.ssh/id_ed25519` is **passphrase-encrypted**
(`ssh-keygen -y` prompts for passphrase; `Load key … incorrect passphrase supplied`).
No passphrase is available on the pod (no .env SSH-related vars, no ssh-agent running,
`/tmp/.env` is the template scaffold synced by provision-pod without an SSH-agent
pass-through). This pod cannot non-interactively SSH to anywhere.

**Decision:** deviate from the spec's storage-pod hand-off. Run all 10 extractions to
the pod-local `/workspace/activations/gemma-3-27b_it/pref_<persona>/` and leave them
there. The workspace volume is a RunPod mfs share (378T, 177T free) so capacity is not
a concern; the user can `rsync` the ~5 GB of outputs to the storage pod manually
once they have a working key. **Skipping the "delete from GPU pod" step** is the only
safe choice — deleting before a verified hand-off would lose the data.

No attempt to provision new infra around this (per "don't work around missing data").
