# 3D MedDiffusion ControlNet Implementation

This document details the implementation of ControlNet for 3D MedDiffusion, enabling conditional generation based on Age and Sex for MRBrain T1/T2 modalities.

## 1. Implementation Overview

The implementation follows the ControlNet paradigm (Zhang et al.) adapted for the specific `BiFlowNet` architecture used in 3D MedDiffusion.

### Architecture (`ddpm/ControlNet.py`)

*   **ControlNet Module**: A trainable copy of the `BiFlowNet` encoder.
    *   **Input**: Takes the noisy latent volume ($x_t$) concatenated with the control volume ($c$) along the channel dimension.
    *   **Encoder Copy**: Includes `init_conv`, `IntraPatchFlow_input` (Transformer blocks), `downs` (ResNet blocks), and `mid_block`.
    *   **Zero Convolutions**: 1x1x1 convolutions initialized to zero are added after every block in the encoder to project features back to the base model's space.
        *   `zero_convs_intra`: For Transformer patch features.
        *   `zero_convs_downs`: For CNN feature maps at multiple resolutions.
        *   `zero_convs_mid`: For the bottleneck features.

*   **ControlledBiFlowNet**: A wrapper class that:
    1.  Freezes the base `BiFlowNet`.
    2.  Runs `ControlNet` to extract feature residuals.
    3.  Injects these residuals into the base `BiFlowNet` decoder (`ups`, `IntraPatchFlow_output`) and middle block.

### BiFlowNet Modifications (`ddpm/BiFlowNet.py`)

*   Modified `forward` method to accept an optional `control_states` dictionary.
*   Added logic to add `control_states` to the skip connections and feature maps in the decoder.

### Conditioning (`dataset/Control_dataset.py`)

*   **Controls**: Age (float 0-1) and Sex (float 0/1).
*   **Spatial Conditioning**: These scalar values are broadcasted to full spatial resolution tensors $(B, 2, D, H, W)$ to match the latent dimensions, preserving spatial correspondence capabilities for future extensions (e.g., segmentation masks).

## 2. Usage Instructions

### Environment
Ensure you are using the project's virtual environment:
```bash
source .venv/bin/activate  # or use .venv/bin/python directly
```

### A. Training Setup

The training process has been updated to support both 8x and 4x models, with optimizations for memory usage and multi-GPU training.

#### 1. Training with 8x Model
For the 8x model, you can train directly from raw images as the memory footprint is manageable.

**Script:** `run_controlnet.sh`
```bash
torchrun --nproc_per_node=4 train/train_ControlNet.py \
  --data-path data/skullstrip/index.json \
  --results-dir results/controlnet_train_8x \
  --pretrained-base-ckpt checkpoints/BiFlowNet_0453500.pt \
  --AE-ckpt checkpoints/PatchVolume_8x_s2.ckpt \
  --resolution 32 32 32 \
  --patch-size 2 \
  --batch-size 1 \
  --num-workers 4 \
  --epochs 100 \
  --downsample-factor 8 \
  --model-dim 72 \
  --dim-mults 1 1 2 4 8 \
  --use-attn 0 0 0 1 1 \
  --volume-channels 8
```

#### 2. Training with 4x Model (Optimized)
Training the 4x model requires significantly more memory. To handle this, we use a two-step process: **pre-computing latents** and then **training on latents**.

**Step 1: Pre-compute Latents**
Run the preprocessing script to encode all images using the 4x AutoEncoder and save them to disk. This supports multi-GPU processing.

```bash
torchrun --nproc_per_node=4 preprocess_latents.py \
  --data-path data/skullstrip/index.json \
  --output-dir data/skullstrip/latents_4x \
  --AE-ckpt checkpoints/PatchVolume4x_s2.ckpt \
  --resolution 48 48 48 \
  --downsample-factor 4
```

**Step 2: Train on Latents**
Use the `run_controlnet_4x.sh` script which points to the pre-computed latents. This bypasses the AE during training, saving memory and time.

```bash
torchrun --nproc_per_node=4 train/train_ControlNet.py \
  --data-path data/skullstrip/index.json \
  --results-dir results/controlnet_train_4x \
  --pretrained-base-ckpt checkpoints/BiFlowNet_4x.pt \
  --AE-ckpt checkpoints/PatchVolume4x_s2.ckpt \
  --latent-root data/skullstrip/latents_4x \
  --resolution 48 48 48 \
  --patch-size 2 \
  --batch-size 1 \
  --downsample-factor 4 \
  ...
```

#### 3. Resuming Training
You can resume training from a checkpoint using the `--ckpt` argument.

**Script:** `run_controlnet_4x_resume.sh`
```bash
torchrun ... \
  --ckpt results/controlnet_train_4x/000-ControlNet/checkpoints/0015000.pt
```

### B. Inference (`inference_ControlNet.py`)

To generate samples with specific Age/Sex conditions:

**Validation Script:** `run_controlnet_val.sh` (matches 4x training resolution)
```bash
python inference_ControlNet.py \
  --base-ckpt checkpoints/BiFlowNet_4x.pt \
  --control-ckpt results/controlnet_train_4x/005-ControlNet/checkpoints/0000100.pt \
  --ae-ckpt checkpoints/PatchVolume4x_s2.ckpt \
  --output-dir results/inference_controlnet_4x \
  --modality T1 \
  --age 0.5 \
  --sex 0.0 \
  --resolution 48 48 48 \
  --timesteps 100
```

## 3. Important Details & Notes

1.  **Memory Management**:
    *   **Pre-computation**: For 4x training, pre-computing latents is crucial to avoid OOM errors and speed up training.
    *   **Distributed Training (DDP)**: Training scripts use `torchrun` for multi-GPU support. `DistributedSampler` ensures data is split correctly.
    *   **Validation**: Validation sampling is only performed on Rank 1 (device 1) to save memory on the primary rank and avoid redundancy.

2.  **Dataset Updates**:
    *   `dataset/Control_dataset.py` now supports:
        *   `latent_root`: Loading `.pt` latent files directly.
        *   `downsample_factor`: Automatically calculating target image size based on latent resolution and factor (4 or 8).
        *   Real metadata: Loads Age and Sex from `index.json`.

3.  **Model Configuration**:
    *   **8x**: Latent resolution 32x32x32 -> Image 256x256x256.
    *   **4x**: Latent resolution 48x48x48 -> Image 192x192x192.
    *   Ensure `resolution`, `patch-size`, and `downsample-factor` match the specific model (4x or 8x) you are using.

## 4. File Structure

*   `ddpm/ControlNet.py`: Core logic for ControlNet and the wrapper.
*   `train/train_ControlNet.py`: Training loop with freezing logic, DDP, and latent support.
*   `preprocess_latents.py`: Script to pre-compute latents for memory-efficient training.
*   `inference_ControlNet.py`: Verification and generation script.
*   `dataset/Control_dataset.py`: Dataset loader with Age/Sex channel generation.
*   `run_controlnet*.sh`: Shell scripts for easy execution of training and validation tasks.
