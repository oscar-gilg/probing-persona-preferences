# Paper-review principles

Distilled from Neel Nanda, *Highly Opinionated Advice on How to Write ML Papers* ([LessWrong](https://www.lesswrong.com/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers)). Each item is reviewable: a subagent reads the relevant scope of the draft and returns a verdict (PASS / WEAK / FAIL) plus quoted evidence.

> **North-star** — every reviewer applies this last: *Does the reader **understand**, **remember**, and **believe** the narrative? If not, which of 1–21 broke?*

## Paper-level (whole draft)

1. **Narrative integrity.** 1–3 specific concrete claims; one cohesive theme. One claim with strong evidence is enough.
2. **Claim calibration.** Each claim's evidential weight is correctly typed: *existence-proof / systematic / hedged-suggestive / narrow / guarantee*. Wording matches the strength of the evidence — no over- or under-claiming.
3. **Belief-update novelty.** *"Should I assign different probabilities to propositions I care about after observing the results?"* Replications, negative results, and rigorous re-checks count as novelty when they update belief; this is acknowledged rather than oversold or hidden.
4. **No verbosity / no jargon-for-impression.** *"Verbosity and overly complex language and jargon is actively detrimental."* Jargon only where it sharpens precision.

## Abstract

5. **Cold-start order.** Uncontroversial field opener → problem/gap → claim with excitement → clarifying detail → evidence sentences (1 per idea) → broader implications. ≥1 concrete metric.

## Introduction

6. **Funnel structure.** Para 1 motivation+citations · Para 2 background · Para 3 main claim with nuance · Para 3.5 critical evidence summary · (optional secondary claims) · Para 4 implications · closing bulleted contributions.
7. **"Why does it matter" is answered.** A reader can articulate stakes from §1 alone.
8. **Liberal citation + honest novelty framing.** Distinguish novel from building-on; cite generously rather than performatively at the end.

## Figures

9. **One takeaway per figure.** Each figure carries a single specific claim; the visual emphasises it (annotations, dark/light contrast for focal element).
10. **Caption tells the story.** Caption alone tells what's plotted, what to compare, and the punchline.
11. **Production hygiene.** Readable titles/legends/axis labels; *"avoid having reds and greens conveying key information, 4% of people are red-green colourblind"*; combine related panels where possible.

## Body (methods / results)

12. **Define before deploy.** *"Please define terminology and crucial techniques!"* (Skip only widely-known terms.)
13. **Methods motivate, not just describe.** Each non-trivial choice has a sentence on *why this approach is relevant to the problem*.
14. **Evidence quality > quantity.** *"Find compelling experiments, not a ton of vaguely relevant ones."* Each headline result has experiments that distinguish between competing hypotheses; ablations one change at a time; strong baselines; multiple qualitatively-different lines of evidence pointing the same way.
15. **Statistical rigor.** Be skeptical of anything not p < .001 in exploratory work — the .005–.05 range has ~28 % replication rate per the literature he cites. Effect sizes reported with noise estimates.
16. **No cherry-picking of qualitative examples.** Random sampling unless making an explicit existence-proof claim.

## Discussion

17. **Limitations stated, not buried.** *"I generally have a much higher opinion of a piece of work if it clearly acknowledges its limitations up front."*
18. **Implications match evidence.** *(Inference from claim calibration.)* Broader-impact claims and future-work bullets stay within what the experiments support.

## Related work (placed at end of paper)

19. **Differentiation, not list.** For each closely-related paper, explain how the contribution differs / replicates / corrects. *"Avoid performatively covering all the relevant papers."* End-placement is consistent with Neel's advice when prior work isn't load-bearing for motivation.

## Appendices & tacit knowledge

20. **Appendix vs main-body fit.** Anything that *carries* a claim is in the body. Support material (full hyperparameters, secondary experiments, replication recipes) lives in the appendix; appendices are *"held to a notably lower standard."*
21. **Tacit knowledge captured.** Implementation challenges, debugging anecdotes, fuzzy intuition, common misconceptions, replication advice — captured in *"appendix A or as an accompanying blog post"*.

---

## How reviewers will use this

- One subagent per **scope** (abstract / intro / figures / body / discussion / related work / appendices).
- Each reviewer applies the principles for its scope plus the four paper-level ones (1–4) plus the north-star.
- Output: per-principle verdict (PASS / WEAK / FAIL) with quoted evidence from the draft, then a prioritised fix list.

## Audit notes (provenance)

- Items 1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21 — direct restatements of Neel's prose.
- Item 8 — refocused on his "liberal citation + replications-count-as-novelty" framing.
- Item 10 — softened ("clear caption" → "caption tells the story") to match his actual wording.
- Item 18 — inference from his claim-calibration emphasis; not a direct quote.
