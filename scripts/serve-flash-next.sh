#!/bin/bash
# Serve Qwen3.8-Flash-Next (tiered experts) with text, image and video input, MTP drafting, on 3x RTX 3090.
# usage: PROFILE=128k|256k LAB=/path/to/lab scripts/serve-flash-next.sh
#   LAB must contain llama.cpp/build/bin (patched build) and models/ (see README).
set -e
LAB=${LAB:-$PWD}; PROFILE=${PROFILE:-128k}; PORT=${PORT:-18100}
MMPROJ=${MMPROJ:-/lab/models/fn-gguf/mmproj-F16.gguf}        # unsloth/Qwen3.8-Flash-Next-GGUF: mmproj-F16.gguf
MMPROJ_DEV=${MMPROJ_DEV:-CUDA1}                              # the card with the most free VRAM; "none" = CPU (~15x slower)
DRAFT=${DRAFT:-/lab/models/fn-gguf/MTP/mtp-shared-iq4.gguf}

if [ "$PROFILE" = 256k ]; then
  MODEL=${MODEL:-/lab/models/tiered/fn-tier50-00001-of-00002.gguf}
  CTX_ARGS="-c 262144 -ctk q8_0 -ctv q8_0 -ts 16,17,15 -np 1"   # 4 slots not tested at 256k
else
  MODEL=${MODEL:-/lab/models/tiered/fn-tier53.gguf}
  # 4 slots sharing one 96k KV pool: an agent's roles (chat, planner, ...) each keep their long prompt prefix
  # cached. With -np 1 every role switch re-reads ~25k tokens (~45 s). 131072 with 4 slots runs out of VRAM.
  CTX_ARGS="-c 98304 -np 4 -kvu -ts 16,16,16"
fi

docker rm -f flash-next >/dev/null 2>&1 || true
docker run -d --name flash-next --restart unless-stopped --gpus all --ipc=host --network host \
  -v "$LAB":/lab -w /lab/llama.cpp/build/bin -e LD_LIBRARY_PATH=/lab/llama.cpp/build/bin \
  lab-cuda:13 \
  ./llama-server -m "$MODEL" --host 0.0.0.0 --port "$PORT" --jinja -fa on --alias qwen3.8-flash-next \
    -ngl 99 $CTX_ARGS -b 1024 -ub 256 -t 4 -tb 28 --metrics \
    --mmproj "$MMPROJ" -mmdev "$MMPROJ_DEV" --image-max-tokens 1024 \
    -md "$DRAFT" --spec-type draft-mtp --spec-draft-n-max 4
echo "flash-next ($PROFILE) starting on :$PORT - first load takes ~3 minutes"
