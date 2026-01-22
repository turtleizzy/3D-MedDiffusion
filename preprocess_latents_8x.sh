#!/bin/bash
source .venv/bin/activate

# Pre-compute latents for 8x model
# Input: Raw images (192x192x192 after crop/pad)
# Output: Latents (24x24x24)

torchrun --nproc_per_node=4 preprocess_latents.py \
  --data-path data/skullstrip/index.json \
  --output-dir data/skullstrip/latents_8x \
  --AE-ckpt checkpoints/PatchVolume_8x_s2.ckpt \
  --resolution 24 24 24 \
  --downsample-factor 8

