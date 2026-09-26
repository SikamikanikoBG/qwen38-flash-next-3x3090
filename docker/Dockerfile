# Build + runtime image for the patched llama.cpp (CUDA 13, sm_86 / RTX 3090).
# ffmpeg is required for video input: llama-server decodes video with ffprobe/ffmpeg, and without them
# a video request fails with HTTP 400 ("ffprobe failed on buffer").
#   docker build -t lab-cuda:13 docker/
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends cmake git libssl-dev libcurl4-openssl-dev ffmpeg \
 && rm -rf /var/lib/apt/lists/*
