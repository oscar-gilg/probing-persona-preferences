#!/bin/bash
# Upload local-only dirs to the storage pod so they can be deleted locally.
set -eu

SSH_HOST='root@213.192.2.99'
SSH_PORT=41560
KEY=~/.ssh/id_ed25519
LOCAL=/Users/oscargilg/Dev/MATS/Preferences/activations
REMOTE=/workspace/activations

ssh $SSH_HOST -p $SSH_PORT -i $KEY -o StrictHostKeyChecking=no "mkdir -p $REMOTE/gemma-3-27b_it $REMOTE/qwen3-emb_8b"

# Transfer locally-only dirs to pod. --archive preserves timestamps. --info=progress2 for a single progress bar.
# 7 dirs, ~15G total.
for dir in \
  gemma-3-27b_it/pref_villain \
  gemma-3-27b_it/pref_midwest \
  gemma-3-27b_it/pref_sadist \
  gemma-3-27b_it/pref_ood_prompts \
  gemma-3-27b_it/pref_ood_category \
  gemma-3-27b_it/truth_error_prefill \
  gemma-3-27b_it/truth_lying_prefill \
  qwen3-emb_8b/pref_main
do
  echo "---> $dir"
  rsync -a --stats -e "ssh -p $SSH_PORT -i $KEY -o StrictHostKeyChecking=no" \
    "$LOCAL/$dir/" "$SSH_HOST:$REMOTE/$dir/" 2>&1 | tail -15
done

echo "Upload complete. Pod gemma-3-27b_it/:"
ssh $SSH_HOST -p $SSH_PORT -i $KEY -o StrictHostKeyChecking=no "du -sh $REMOTE/gemma-3-27b_it/* | sort -h"
