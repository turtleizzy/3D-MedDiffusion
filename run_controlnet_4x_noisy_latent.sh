#!/bin/bash
source .venv/bin/activate

# Training ControlNet with Noisy Latent Control on skullstrip data
# Using 4x AE and 4x Base Model
# Resolution 48x48x48 (Latent) -> 192x192x192 (Image)
# Control channels: 2 (age, sex) + 8 (noisy latent) = 10
#
# This script:
# 1. Uses pre-computed latents from data/skullstrip/latents_4x
# 2. Adds noisy latent as additional control signal
# 3. Noise strength randomly sampled [0, max_noise_strength] per sample

torchrun --nproc_per_node=4 train/train_ControlNet.py \
  --data-path data/skullstrip/index.json \
  --results-dir results/controlnet_train_4x_noisy_latent \
  --pretrained-base-ckpt checkpoints/BiFlowNet_4x.pt \
  --AE-ckpt checkpoints/PatchVolume4x_s2.ckpt \
  --resolution 48 48 48 \
  --patch-size 2 \
  --batch-size 4 \
  --num-workers 4 \
  --downsample-factor 4 \
  --epochs 300 \
  --log-every 5 \
  --ckpt-every 1000 \
  --model-dim 72 \
  --dim-mults 1 1 2 4 8 \
  --use-attn 0 0 0 1 1 \
  --volume-channels 8 \
  --latent-root data/skullstrip/latents_4x \
  --use-noisy-latent-control \
  --max-noise-strength 1.0 \
  --verify-data

