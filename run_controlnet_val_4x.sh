#!/bin/bash
source .venv/bin/activate

# Training ControlNet on skullstrip data (MRTIBrain task)
# Using 8x AE and 8x Base Model (BiFlowNet_0453500.pt)
# Resolution 24x24x24 (Latent) -> 192x192x192 (Image)
# Patch size 1 for 8x model

.venv/bin/python inference_ControlNet.py \
  --base-ckpt checkpoints/BiFlowNet_4x.pt \
  --control-ckpt results/controlnet_train_4x/000-ControlNet/checkpoints/0005000.pt \
  --ae-ckpt checkpoints/PatchVolume4x_s2.ckpt \
  --output-dir results/inference_controlnet_4x \
  --modality T1 \
  --age 0.8 \
  --sex 0.0 \
  --resolution 48 48 48 \
  --patch-size 2 \
  --timesteps 1000