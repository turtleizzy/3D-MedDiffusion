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
RESULTS_DIR="results/controlnet_train_4x_augmented_alternating_fixed_latent"
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

# Validation parameters
VAL_DIR="data/validation"  # Directory containing .nii.gz files for validation
VAL_SEX=0.0
VAL_NOISE_STRENGTH=0.5
VAL_AGES="0.2 0.8"  # Age values to test for each file

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
        
        # Build base inference command
        INFERENCE_CMD=".venv/bin/python inference_ControlNet.py"
        INFERENCE_CMD="${INFERENCE_CMD} --base-ckpt ${PRETRAINED_BASE_CKPT}"
        INFERENCE_CMD="${INFERENCE_CMD} --control-ckpt \"${LATEST_CKPT}\""
        INFERENCE_CMD="${INFERENCE_CMD} --ae-ckpt ${AE_CKPT}"
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
        
        # Scan validation directory for all .nii.gz files
        if [ ! -d "${VAL_DIR}" ]; then
            echo "Warning: Validation directory ${VAL_DIR} does not exist. Skipping validation."
        else
            # Find all .nii.gz files in the validation directory
            VAL_FILES=$(find "${VAL_DIR}" -name "*.nii.gz" -type f | sort)
            
            # Count files (handle empty result)
            if [ -z "${VAL_FILES}" ]; then
                VAL_FILE_COUNT=0
            else
                VAL_FILE_COUNT=$(echo "${VAL_FILES}" | grep -c .)
            fi
            
            if [ "${VAL_FILE_COUNT}" -eq 0 ]; then
                echo "Warning: No .nii.gz files found in ${VAL_DIR}. Skipping validation."
            else
                echo "Found ${VAL_FILE_COUNT} .nii.gz files in ${VAL_DIR}"
                
                # Count number of age values
                AGE_COUNT=0
                for age in ${VAL_AGES}; do
                    AGE_COUNT=$((AGE_COUNT + 1))
                done
                
                # Calculate total cases: files with latent + cases without latent
                CASES_WITH_LATENT=$((VAL_FILE_COUNT * AGE_COUNT))
                CASES_WITHOUT_LATENT=${AGE_COUNT}  # One case per age value without latent
                TOTAL_CASES=$((CASES_WITH_LATENT + CASES_WITHOUT_LATENT))
                
                echo "Generating predictions:"
                echo "  - ${CASES_WITH_LATENT} cases with latent input (${VAL_FILE_COUNT} files × ${AGE_COUNT} ages)"
                echo "  - ${CASES_WITHOUT_LATENT} cases without latent input (${AGE_COUNT} ages)"
                echo "  Total: ${TOTAL_CASES} cases"
                
                FILE_INDEX=0
                CURRENT_CASE=0
                
                # Process each file
                while IFS= read -r VAL_NIFTI; do
                    if [ -z "${VAL_NIFTI}" ]; then
                        continue
                    fi
                    
                    FILE_INDEX=$((FILE_INDEX + 1))
                    FILENAME=$(basename "${VAL_NIFTI}" .nii.gz)
                    
                    # Create per-file output directory
                    FILE_OUTPUT_DIR="${VAL_OUTPUT_DIR}/${FILENAME}"
                    mkdir -p "${FILE_OUTPUT_DIR}"
                    
                    # Generate predictions for each age value
                    for VAL_AGE in ${VAL_AGES}; do
                        CURRENT_CASE=$((CURRENT_CASE + 1))
                        echo ""
                        echo "Running validation case ${CURRENT_CASE}/${TOTAL_CASES}:"
                        echo "  File: ${FILENAME} (${FILE_INDEX}/${VAL_FILE_COUNT})"
                        echo "  Age: ${VAL_AGE}"
                        
                        # Build command for this specific case
                        CASE_CMD="${INFERENCE_CMD}"
                        CASE_CMD="${CASE_CMD} --output-dir ${FILE_OUTPUT_DIR}"
                        CASE_CMD="${CASE_CMD} --age ${VAL_AGE}"
                        CASE_CMD="${CASE_CMD} --input-nifti \"${VAL_NIFTI}\""
                        
                        # Execute inference
                        eval ${CASE_CMD}
                        
                        if [ $? -ne 0 ]; then
                            echo "Warning: Validation failed for ${FILENAME} with age=${VAL_AGE}"
                        else
                            echo "Successfully generated prediction for ${FILENAME} with age=${VAL_AGE}"
                        fi
                    done
                done <<< "${VAL_FILES}"
                
                # Generate predictions without latent input
                echo ""
                echo "=========================================="
                echo "Generating predictions without latent input"
                echo "=========================================="
                
                NO_LATENT_OUTPUT_DIR="${VAL_OUTPUT_DIR}/no_latent_input"
                mkdir -p "${NO_LATENT_OUTPUT_DIR}"
                
                for VAL_AGE in ${VAL_AGES}; do
                    CURRENT_CASE=$((CURRENT_CASE + 1))
                    echo ""
                    echo "Running validation case ${CURRENT_CASE}/${TOTAL_CASES}:"
                    echo "  No latent input"
                    echo "  Age: ${VAL_AGE}"
                    
                    # Build command without input-nifti
                    CASE_CMD="${INFERENCE_CMD}"
                    CASE_CMD="${CASE_CMD} --output-dir ${NO_LATENT_OUTPUT_DIR}"
                    CASE_CMD="${CASE_CMD} --age ${VAL_AGE}"
                    # Note: No --input-nifti flag, so no latent input
                    
                    # Execute inference
                    eval ${CASE_CMD}
                    
                    if [ $? -ne 0 ]; then
                        echo "Warning: Validation failed for no latent input with age=${VAL_AGE}"
                    else
                        echo "Successfully generated prediction without latent input for age=${VAL_AGE}"
                    fi
                done
                
                echo ""
                echo "Validation completed."
                echo "  - Processed ${FILE_INDEX} files with latent input (${AGE_COUNT} age values each)"
                echo "  - Generated ${CASES_WITHOUT_LATENT} predictions without latent input"
                echo "Results saved to ${VAL_OUTPUT_DIR}"
            fi
        fi
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

