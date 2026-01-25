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

echo "Augmentation completed. Checking generated latents and extracting sample IDs..."
# Verify latent dimensions and extract actual sample stems
python -c "
import torch
import glob
import os
import re

latent_dir = '${LATENT_ROOT}'
aug_files = glob.glob(os.path.join(latent_dir, '*_aug*.pt'))
print(f'Found {len(aug_files)} augmented latent files...')

# Extract unique stems from augmented latent filenames
stems = set()
for f in aug_files:
    basename = os.path.basename(f)
    # Extract stem: remove _aug*.pt suffix
    match = re.match(r'(.+?)_aug\d+\.pt$', basename)
    if match:
        stem = match.group(1)
        stems.add(stem)
    else:
        print(f'Warning: Unexpected filename format: {basename}')

print(f'Found {len(stems)} unique samples with augmented latents:')
for stem in sorted(stems):
    print(f'  - {stem}')

# Verify latent dimensions
print(f'\\nChecking latent dimensions...')
checked = 0
for f in aug_files[:10]:  # Check first 10 files
    data = torch.load(f)
    expected_shape = (8, 48, 48, 48)
    if data.shape == expected_shape:
        checked += 1
    else:
        print(f'✗ {os.path.basename(f)}: {data.shape} (expected {expected_shape})')
        exit(1)
print(f'✓ Checked {checked} latents, all have correct dimensions!')

# Save stems to a file for use in next step
stems_file = os.path.join(latent_dir, '.generated_stems.txt')
with open(stems_file, 'w') as f:
    for stem in sorted(stems):
        f.write(stem + '\\n')
print(f'\\nSaved {len(stems)} sample stems to {stems_file}')
"

if [ $? -ne 0 ]; then
    echo "Error: Latent dimension verification failed."
    exit 1
fi

# Step 2: Training for 1 checkpoint cycle
echo ""
echo "Step 2: Training for ${CKPT_CYCLES} checkpoint cycle(s)..."

# First, use jq to inspect the structure of index.json
echo "Inspecting index.json structure..."
INDEX_STRUCTURE=$(jq '.[0]' "${DATA_PATH}" 2>/dev/null)
if [ $? -ne 0 ] || [ -z "$INDEX_STRUCTURE" ]; then
    echo "Error: Failed to read index.json structure using jq"
    exit 1
fi
echo "First entry structure:"
echo "$INDEX_STRUCTURE" | head -10
echo ""

# Create a subset index.json for training based on actually generated latents
python -c "
import json
import os
import re
import subprocess

# Load original index
with open('${DATA_PATH}', 'r') as f:
    data = json.load(f)

# Verify structure using jq
try:
    result = subprocess.run(['jq', '.[0]', '${DATA_PATH}'], 
                          capture_output=True, text=True, check=True)
    first_entry = json.loads(result.stdout)
    print(f'Verified index.json structure. First entry keys: {list(first_entry.keys())}')
    
    # Check if studyUID exists
    if 'studyUID' not in first_entry:
        print('Warning: studyUID not found in first entry. Available keys:', list(first_entry.keys()))
except Exception as e:
    print(f'Warning: Could not verify structure with jq: {e}')

# Load stems from generated latents
stems_file = os.path.join('${LATENT_ROOT}', '.generated_stems.txt')
if not os.path.exists(stems_file):
    print(f'Error: Stems file not found: {stems_file}')
    exit(1)

with open(stems_file, 'r') as f:
    generated_stems = set(line.strip() for line in f if line.strip())

print(f'Loaded {len(generated_stems)} generated stems from {stems_file}')

# Filter index.json to only include samples that have generated latents
# index.json is a list of objects, each with studyUID field
# studyUID is the first part of the filename (e.g., '7BB82F5707974FFEB60B838AD0AF5A31')
# Generated latent filenames start with studyUID
test_data = []
for item in data:
    # Extract studyUID from item
    study_uid = item.get('studyUID')
    if not study_uid:
        print(f'Warning: Entry missing studyUID: {item}')
        continue
    
    # Check if any generated stem starts with this studyUID
    # Generated stems are full filenames without extension, e.g., '7BB82F5707974FFEB60B838AD0AF5A31_xxx_mni'
    matched = False
    for stem in generated_stems:
        # stem should start with studyUID
        if stem.startswith(study_uid):
            matched = True
            break
    
    if matched:
        test_data.append(item)

print(f'Filtered index.json: {len(data)} -> {len(test_data)} samples')
if len(test_data) == 0:
    print('Error: No matching samples found!')
    print('Generated stems (first 5):')
    for stem in list(generated_stems)[:5]:
        print(f'  - {stem}')
    print('First entry studyUID:', data[0].get('studyUID') if data else 'N/A')
    exit(1)

# Save test index
test_index_path = 'data/skullstrip/index_test.json'
os.makedirs(os.path.dirname(test_index_path), exist_ok=True)
with open(test_index_path, 'w') as f:
    json.dump(test_data, f, indent=2)

print(f'Created test index with {len(test_data)} samples: {test_index_path}')
"

if [ $? -ne 0 ]; then
    echo "Error: Failed to create test index based on generated latents."
    exit 1
fi

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

