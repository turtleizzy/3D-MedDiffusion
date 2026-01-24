#!/bin/bash
source .venv/bin/activate

# Validation script for ControlNet with Noisy Latent Control
# Using 4x AE and 4x Base Model
# Resolution 48x48x48 (Latent) -> 192x192x192 (Image)
#
# Usage examples:
#   1. Without input (age/sex only): bash run_controlnet_val_4x_noisy_latent.sh
#   2. With NIfTI input: bash run_controlnet_val_4x_noisy_latent.sh --input-nifti /path/to/input.nii.gz
#   3. With pre-computed latent: bash run_controlnet_val_4x_noisy_latent.sh --input-latent /path/to/latent.pt
#   4. With custom noise strength: bash run_controlnet_val_4x_noisy_latent.sh --input-nifti /path/to/input.nii.gz --noise-strength 0.3

# Default values
CONTROL_CKPT="results/controlnet_train_4x_augmented_alternating/001-ControlNet_NoisyLatent/checkpoints/0005000.pt"
OUTPUT_DIR="results/inference_controlnet_4x_noisy_latent_alternating_ckpt5000"
AGE=0.8
SEX=0.0
NOISE_STRENGTH=0.5
INPUT_NIFTI= #"data/UCSD-PTGBM/T1Pre/UCSD-PTGBM-0002_01_T1pre.nii.gz"

.venv/bin/python inference_ControlNet.py \
  --base-ckpt checkpoints/BiFlowNet_4x.pt \
  --control-ckpt ${CONTROL_CKPT} \
  --ae-ckpt checkpoints/PatchVolume4x_s2.ckpt \
  --output-dir ${OUTPUT_DIR} \
  --modality T1 \
  --age ${AGE} \
  --sex ${SEX} \
  --resolution 48 48 48 \
  --patch-size 2 \
  --timesteps 1000 \
  --downsample-factor 4 \
  --use-noisy-latent-control \
  # --input-nifti ${INPUT_NIFTI} \
  # --noise-strength ${NOISE_STRENGTH} \
  # "$@"

