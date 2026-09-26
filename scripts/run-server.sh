#!/bin/bash
# usage: run-server.sh <name> <model.gguf> [extra llama-server args...]
LAB=${LAB:-$PWD}
name=$1; model=$2; shift 2
docker rm -f $name >/dev/null 2>&1
docker run -d --name $name --gpus all --ipc=host --network host \
  -v "$LAB":/lab -w /lab/llama.cpp/build/bin \
  -e LD_LIBRARY_PATH=/lab/llama.cpp/build/bin ${DOCKER_EXTRA:-} \
  lab-cuda:13 \
  ./llama-server -m "$model" --host 0.0.0.0 --port ${PORT:-18100} --jinja -fa on "$@"
