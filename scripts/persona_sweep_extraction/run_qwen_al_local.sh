#!/bin/bash
# Launch the 21 Qwen persona-sweep AL runs locally. Hits OpenRouter for Qwen
# completions + gpt-5-nano judge parses. Tmux-wrapped for SSH-drop survival.
#
# SSL cert exports below sidestep a ~30 min startup hang in
# ssl.SSLContext.load_verify_locations under many concurrent clients.

set -u
set -o pipefail

cd /Users/oscargilg/Dev/MATS/Preferences

set -a
source .env
set +a

export SSL_CERT_FILE=$(python -m certifi)
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE

mkdir -p logs

python -m src.measurement.runners.run \
    configs/measurement/qwen_persona_sweep/final_six/aura_eval.yaml \
    configs/measurement/qwen_persona_sweep/final_six/aura_test.yaml \
    configs/measurement/qwen_persona_sweep/final_six/aura_train.yaml \
    configs/measurement/qwen_persona_sweep/final_six/contrarian_eval.yaml \
    configs/measurement/qwen_persona_sweep/final_six/contrarian_test.yaml \
    configs/measurement/qwen_persona_sweep/final_six/contrarian_train.yaml \
    configs/measurement/qwen_persona_sweep/final_six/default_eval.yaml \
    configs/measurement/qwen_persona_sweep/final_six/default_test.yaml \
    configs/measurement/qwen_persona_sweep/final_six/default_train.yaml \
    configs/measurement/qwen_persona_sweep/final_six/mathematician_eval.yaml \
    configs/measurement/qwen_persona_sweep/final_six/mathematician_test.yaml \
    configs/measurement/qwen_persona_sweep/final_six/mathematician_train.yaml \
    configs/measurement/qwen_persona_sweep/final_six/sadist_eval.yaml \
    configs/measurement/qwen_persona_sweep/final_six/sadist_test.yaml \
    configs/measurement/qwen_persona_sweep/final_six/sadist_train.yaml \
    configs/measurement/qwen_persona_sweep/final_six/slacker_eval.yaml \
    configs/measurement/qwen_persona_sweep/final_six/slacker_test.yaml \
    configs/measurement/qwen_persona_sweep/final_six/slacker_train.yaml \
    configs/measurement/qwen_persona_sweep/final_six/strategist_eval.yaml \
    configs/measurement/qwen_persona_sweep/final_six/strategist_test.yaml \
    configs/measurement/qwen_persona_sweep/final_six/strategist_train.yaml \
    --max-concurrent 50 \
    --experiment-id qwen_persona_sweep_final_six 2>&1 | tee logs/qwen_persona_sweep_final_six.log
