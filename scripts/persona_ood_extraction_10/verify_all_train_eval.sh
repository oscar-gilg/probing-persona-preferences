#!/bin/bash
# Run verify_extraction_train_eval.py on all 10 personas.
set -u
REPO=/workspace/repo
PERSONAS=(
    evil_genius chaos_agent obsessive_perfectionist lazy_minimalist
    nationalist_ideologue conspiracy_theorist contrarian_intellectual
    whimsical_poet depressed_nihilist people_pleaser
)
cd "$REPO"
for p in "${PERSONAS[@]}"; do
    echo "=== $p ==="
    python scripts/persona_ood_extraction_10/verify_extraction_train_eval.py "$p" || echo "FAILED $p"
    echo
done
