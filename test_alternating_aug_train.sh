#!/bin/bash
# Test script for alternating augmentation and training with small dataset
# This script uses a small subset of data to verify the training pipeline works correctly

source .venv/bin/activate

# Configuration - using small values for testing
DATA_PATH="data/skullstrip/index.json"
LATENT_ROOT="data/skullstrip/latents_4x_augmented_test"
AE_CKPT="checkpoints/PatchVolume4x_s2.ckpt"
RESULTS_DIR="results/controlnet_train_4x_augmented_test"
PRETRAINED_BASE_CKPT="checkpoints/BiFlowNet_4x.pt"
RESOLUTION="48 48 48"
DOWNSAMPLE_FACTOR=4
NUM_AUGMENTATIONS=2  # Reduced for testing
CKPT_CYCLES=1  # Only 1 checkpoint cycle for testing
CKPT_EVERY=10  # Very small for quick testing
BATCH_SIZE=2  # Smaller batch size
NUM_WORKERS=2  # Fewer workers for testing

# Training parameters
MODEL_DIM=72
DIM_MULTS="1 1 2 4 8"
USE_ATTN="0 0 0 1 1"
PATCH_SIZE=2
VOLUME_CHANNELS=8
EPOCHS=1  # Only 1 epoch for testing
LOG_EVERY=2  # Log more frequently for testing
FULL_NOISE_PROB=0.5

# Create directories
mkdir -p "${LATENT_ROOT}"
mkdir -p "${RESULTS_DIR}"

echo "=========================================="
echo "TEST: Alternating Augmentation and Training"
echo "=========================================="
echo "Latent root: ${LATENT_ROOT}"
echo "Results dir: ${RESULTS_DIR}"
echo "Checkpoint cycles: ${CKPT_CYCLES}"
echo "Checkpoint every: ${CKPT_EVERY} steps"
echo "Epochs: ${EPOCHS}"
echo ""

# Step 1: Generate augmented latents for a small subset
echo "Step 1: Generating augmented latents for first 5 samples..."
torchrun --nproc_per_node=2 preprocess_latents_augmented.py \
  --data-path "${DATA_PATH}" \
  --output-dir "${LATENT_ROOT}" \
  --AE-ckpt "${AE_CKPT}" \
  --resolution ${RESOLUTION} \
  --downsample-factor ${DOWNSAMPLE_FACTOR} \
  --num-augmentations ${NUM_AUGMENTATIONS} \
  --seed 42 \
  --overwrite \
  --max-samples 5

if [ $? -ne 0 ]; then
    echo "Error: Augmentation failed. Stopping."
    exit 1
fi

echo "Augmentation completed. Checking generated latents..."
# Verify latent dimensions
python -c "
import torch
import glob
import os

latent_dir = '${LATENT_ROOT}'
files = glob.glob(os.path.join(latent_dir, '*_aug*.pt'))[:5]
print(f'Checking {len(files)} augmented latents...')
for f in files:
    data = torch.load(f)
    expected_shape = (8, 48, 48, 48)
    if data.shape == expected_shape:
        print(f'✓ {os.path.basename(f)}: {data.shape} (correct)')
    else:
        print(f'✗ {os.path.basename(f)}: {data.shape} (expected {expected_shape})')
        exit(1)
print('All latents have correct dimensions!')
"

if [ $? -ne 0 ]; then
    echo "Error: Latent dimension verification failed."
    exit 1
fi

# Step 2: Training for 1 checkpoint cycle
echo ""
echo "Step 2: Training for ${CKPT_CYCLES} checkpoint cycle(s)..."

# Create a small subset index.json for training
python -c "
import json
import os

# Load original index
with open('${DATA_PATH}', 'r') as f:
    data = json.load(f)

# Take first 5 samples
test_data = data[:5]

# Save test index
test_index_path = 'data/skullstrip/index_test.json'
os.makedirs(os.path.dirname(test_index_path), exist_ok=True)
with open(test_index_path, 'w') as f:
    json.dump(test_data, f, indent=2)

print(f'Created test index with {len(test_data)} samples: {test_index_path}')
"

# Build training command
TRAIN_CMD="torchrun --nproc_per_node=2 train/train_ControlNet.py"
TRAIN_CMD="${TRAIN_CMD} --data-path data/skullstrip/index_test.json"
TRAIN_CMD="${TRAIN_CMD} --results-dir ${RESULTS_DIR}"
TRAIN_CMD="${TRAIN_CMD} --pretrained-base-ckpt ${PRETRAINED_BASE_CKPT}"
TRAIN_CMD="${TRAIN_CMD} --AE-ckpt ${AE_CKPT}"
TRAIN_CMD="${TRAIN_CMD} --resolution ${RESOLUTION}"
TRAIN_CMD="${TRAIN_CMD} --patch-size ${PATCH_SIZE}"
TRAIN_CMD="${TRAIN_CMD} --batch-size ${BATCH_SIZE}"
TRAIN_CMD="${TRAIN_CMD} --num-workers ${NUM_WORKERS}"
TRAIN_CMD="${TRAIN_CMD} --downsample-factor ${DOWNSAMPLE_FACTOR}"
TRAIN_CMD="${TRAIN_CMD} --epochs ${EPOCHS}"
TRAIN_CMD="${TRAIN_CMD} --log-every ${LOG_EVERY}"
TRAIN_CMD="${TRAIN_CMD} --ckpt-every ${CKPT_EVERY}"
TRAIN_CMD="${TRAIN_CMD} --ckpt-cycles ${CKPT_CYCLES}"
TRAIN_CMD="${TRAIN_CMD} --model-dim ${MODEL_DIM}"
TRAIN_CMD="${TRAIN_CMD} --dim-mults ${DIM_MULTS}"
TRAIN_CMD="${TRAIN_CMD} --use-attn ${USE_ATTN}"
TRAIN_CMD="${TRAIN_CMD} --volume-channels ${VOLUME_CHANNELS}"
TRAIN_CMD="${TRAIN_CMD} --latent-root ${LATENT_ROOT}"
TRAIN_CMD="${TRAIN_CMD} --use-noisy-latent-control"
TRAIN_CMD="${TRAIN_CMD} --max-noise-strength 1.0"
TRAIN_CMD="${TRAIN_CMD} --full-noise-prob ${FULL_NOISE_PROB}"
TRAIN_CMD="${TRAIN_CMD} --verify-data"  # Verify data shapes before training

echo "Starting training from scratch"
echo "Command: ${TRAIN_CMD}"
echo ""

# Execute training
eval ${TRAIN_CMD}

if [ $? -ne 0 ]; then
    echo "Error: Training failed. Stopping."
    exit 1
fi

echo ""
echo "=========================================="
echo "TEST COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo "Checkpoints saved to: ${RESULTS_DIR}"
echo "Augmented latents in: ${LATENT_ROOT}"
echo ""

