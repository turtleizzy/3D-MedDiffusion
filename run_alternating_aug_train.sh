#!/bin/bash
# Alternating augmentation and training cycle script
# This script alternates between:
# 1. Generating augmented latents (10 versions per sample)
# 2. Training for 2 checkpoint cycles
# 3. Repeating the cycle

source .venv/bin/activate

# Configuration
DATA_PATH="data/skullstrip/index.json"
LATENT_ROOT="data/skullstrip/latents_4x_augmented"
AE_CKPT="checkpoints/PatchVolume4x_s2.ckpt"
RESULTS_DIR="results/controlnet_train_4x_augmented_alternating"
PRETRAINED_BASE_CKPT="checkpoints/BiFlowNet_4x.pt"
RESOLUTION="48 48 48"
DOWNSAMPLE_FACTOR=4
NUM_AUGMENTATIONS=3
CKPT_CYCLES=5
CKPT_EVERY=1000
BATCH_SIZE=4
NUM_WORKERS=4

# Training parameters
MODEL_DIM=72
DIM_MULTS="1 1 2 4 8"
USE_ATTN="0 0 0 1 1"
PATCH_SIZE=2
VOLUME_CHANNELS=8
EPOCHS=300
LOG_EVERY=5
FULL_NOISE_PROB=0.5  # 50% probability of using full noise

# Create directories
mkdir -p "${LATENT_ROOT}"
mkdir -p "${RESULTS_DIR}"

# Initialize checkpoint path (will be updated in loop)
CURRENT_CKPT=

# Cycle counter
CYCLE=0
MAX_CYCLES=100  # Maximum number of cycles (set to large number for continuous training)

echo "Starting alternating augmentation and training cycle"
echo "Latent root: ${LATENT_ROOT}"
echo "Results dir: ${RESULTS_DIR}"
echo "Checkpoint cycles per training: ${CKPT_CYCLES}"
echo "Full noise probability: ${FULL_NOISE_PROB}"

while [ $CYCLE -lt $MAX_CYCLES ]; do
    CYCLE=$((CYCLE + 1))
    echo ""
    echo "=========================================="
    echo "Cycle ${CYCLE}"
    echo "=========================================="
    
    # Step 1: Generate augmented latents
    echo ""
    echo "Step 1: Generating augmented latents..."
    echo "This will generate ${NUM_AUGMENTATIONS} augmented versions per sample"

    if [ $CYCLE -le 0 ]; then
        echo "Skipping augmentation for cycle ${CYCLE}"
    else
        torchrun --nproc_per_node=4 preprocess_latents_augmented.py \
        --data-path "${DATA_PATH}" \
        --output-dir "${LATENT_ROOT}" \
        --AE-ckpt "${AE_CKPT}" \
        --resolution ${RESOLUTION} \
        --downsample-factor ${DOWNSAMPLE_FACTOR} \
        --num-augmentations ${NUM_AUGMENTATIONS} \
        --seed $((42 + CYCLE)) \
        --overwrite
        
        if [ $? -ne 0 ]; then
            echo "Error: Augmentation failed. Stopping."
            exit 1
        fi
    fi
    
    echo "Augmentation completed. Latents saved to ${LATENT_ROOT}"
    
    # Step 2: Training for 2 checkpoint cycles
    echo ""
    echo "Step 2: Training for ${CKPT_CYCLES} checkpoint cycles..."
    
    # Build training command
    TRAIN_CMD="torchrun --nproc_per_node=4 train/train_ControlNet.py"
    TRAIN_CMD="${TRAIN_CMD} --data-path ${DATA_PATH}"
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
    
    # Add checkpoint path if available
    if [ -n "${CURRENT_CKPT}" ] && [ -f "${CURRENT_CKPT}" ]; then
        TRAIN_CMD="${TRAIN_CMD} --ckpt \"${CURRENT_CKPT}\""
        echo "Resuming from checkpoint: ${CURRENT_CKPT}"
    else
        echo "Starting training from scratch"
    fi
    
    # Execute training
    eval ${TRAIN_CMD}
    
    if [ $? -ne 0 ]; then
        echo "Error: Training failed. Stopping."
        exit 1
    fi
    
    echo "Training completed for ${CKPT_CYCLES} checkpoint cycles"
    
    # Find the latest checkpoint
    # Checkpoints are saved as: results_dir/XXX-ControlNet_NoisyLatent/checkpoints/XXXXXXX.pt
    LATEST_CKPT=$(find "${RESULTS_DIR}" -name "*.pt" -path "*/checkpoints/*" -type f | sort -V | tail -1)
    
    if [ -n "${LATEST_CKPT}" ] && [ -f "${LATEST_CKPT}" ]; then
        CURRENT_CKPT="${LATEST_CKPT}"
        echo "Latest checkpoint: ${CURRENT_CKPT}"
        
        # Step 3: Run validation with latest checkpoint
        echo ""
        echo "Step 3: Running validation with latest checkpoint..."
        
        # Create validation output directory
        VAL_OUTPUT_DIR="${RESULTS_DIR}/validation_cycle${CYCLE}"
        mkdir -p "${VAL_OUTPUT_DIR}"
        
        # Validation parameters
        VAL_NIFTI="data/UCSD-PTGBM/UCSD-PTGBM-0166_01_T1pre_masked.nii.gz"
        VAL_SEX=0.0
        VAL_NOISE_STRENGTH=0.5
        
        # Build base inference command
        INFERENCE_CMD=".venv/bin/python inference_ControlNet.py"
        INFERENCE_CMD="${INFERENCE_CMD} --base-ckpt ${PRETRAINED_BASE_CKPT}"
        INFERENCE_CMD="${INFERENCE_CMD} --control-ckpt \"${LATEST_CKPT}\""
        INFERENCE_CMD="${INFERENCE_CMD} --ae-ckpt ${AE_CKPT}"
        INFERENCE_CMD="${INFERENCE_CMD} --output-dir ${VAL_OUTPUT_DIR}"
        INFERENCE_CMD="${INFERENCE_CMD} --modality T1"
        INFERENCE_CMD="${INFERENCE_CMD} --sex ${VAL_SEX}"
        INFERENCE_CMD="${INFERENCE_CMD} --resolution ${RESOLUTION}"
        INFERENCE_CMD="${INFERENCE_CMD} --patch-size ${PATCH_SIZE}"
        INFERENCE_CMD="${INFERENCE_CMD} --timesteps 1000"
        INFERENCE_CMD="${INFERENCE_CMD} --downsample-factor ${DOWNSAMPLE_FACTOR}"
        INFERENCE_CMD="${INFERENCE_CMD} --use-noisy-latent-control"
        INFERENCE_CMD="${INFERENCE_CMD} --model-dim ${MODEL_DIM}"
        INFERENCE_CMD="${INFERENCE_CMD} --dim-mults ${DIM_MULTS}"
        INFERENCE_CMD="${INFERENCE_CMD} --volume-channels ${VOLUME_CHANNELS}"
        INFERENCE_CMD="${INFERENCE_CMD} --use-attn ${USE_ATTN}"
        INFERENCE_CMD="${INFERENCE_CMD} --noise-strength ${VAL_NOISE_STRENGTH}"
        
        # Run 4 validation cases
        echo "Running validation case 1/4: No nifti, age=0.2"
        eval ${INFERENCE_CMD} --age 0.2
        
        if [ $? -ne 0 ]; then
            echo "Warning: Validation case 1 failed."
        fi
        
        echo "Running validation case 2/4: No nifti, age=0.8"
        eval ${INFERENCE_CMD} --age 0.8
        
        if [ $? -ne 0 ]; then
            echo "Warning: Validation case 2 failed."
        fi
        
        echo "Running validation case 3/4: With nifti, age=0.2"
        eval ${INFERENCE_CMD} --age 0.2 --input-nifti "${VAL_NIFTI}"
        
        if [ $? -ne 0 ]; then
            echo "Warning: Validation case 3 failed."
        fi
        
        echo "Running validation case 4/4: With nifti, age=0.8"
        eval ${INFERENCE_CMD} --age 0.8 --input-nifti "${VAL_NIFTI}"
        
        if [ $? -ne 0 ]; then
            echo "Warning: Validation case 4 failed."
        fi
        
        echo "Validation completed. Results saved to ${VAL_OUTPUT_DIR}"
    else
        echo "Warning: No checkpoint found. Will start from scratch in next cycle."
        CURRENT_CKPT=""
    fi
    
    
    echo ""
    echo "Cycle ${CYCLE} completed. Waiting before next cycle..."
    sleep 5
done

echo ""
echo "All cycles completed!"

