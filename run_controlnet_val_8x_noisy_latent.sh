#!/bin/bash
source .venv/bin/activate

# Validation script for ControlNet with Noisy Latent Control
# Using 8x AE and 8x Base Model
# Resolution 24x24x24 (Latent) -> 192x192x192 (Image)
#
# Usage examples:
#   1. Without input (age/sex only): bash run_controlnet_val_8x_noisy_latent.sh
#   2. With NIfTI input: bash run_controlnet_val_8x_noisy_latent.sh --input-nifti /path/to/input.nii.gz
#   3. With pre-computed latent: bash run_controlnet_val_8x_noisy_latent.sh --input-latent /path/to/latent.pt
#   4. With custom noise strength: bash run_controlnet_val_8x_noisy_latent.sh --input-nifti /path/to/input.nii.gz --noise-strength 0.3

# Default values
CONTROL_CKPT="results/controlnet_train_8x_noisy_latent/001-ControlNet_NoisyLatent/checkpoints/0000050.pt"
OUTPUT_DIR="results/inference_controlnet_8x_noisy_latent"
AGE=0.5
SEX=0.0
NOISE_STRENGTH=0.5

.venv/bin/python inference_ControlNet.py \
  --base-ckpt checkpoints/BiFlowNet_0453500.pt \
  --control-ckpt ${CONTROL_CKPT} \
  --ae-ckpt checkpoints/PatchVolume_8x_s2.ckpt \
  --output-dir ${OUTPUT_DIR} \
  --modality T1 \
  --age ${AGE} \
  --sex ${SEX} \
  --resolution 24 24 24 \
  --patch-size 1 \
  --timesteps 1000 \
  --downsample-factor 8 \
  --use-noisy-latent-control \
  --noise-strength ${NOISE_STRENGTH} \
  "$@"

