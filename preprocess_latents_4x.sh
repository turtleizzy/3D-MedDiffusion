#!/bin/bash
source .venv/bin/activate

# Pre-compute latents for 8x model
# Input: Raw images (192x192x192 after crop/pad)

torchrun --nproc_per_node=1 preprocess_latents.py \
  --data-path data/skullstrip/index.json \
  --output-dir data/skullstrip/latents_4x \
  --AE-ckpt checkpoints/PatchVolume4x_s2.ckpt \
  --resolution 48 48 48 \
  --downsample-factor 4

