import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "."))
sys.path.append(project_root)
import torch
import torch.distributed as dist
import argparse
import logging
import numpy as np
from AutoEncoder.model.PatchVolume import patchvolumeAE
from dataset.Control_dataset import Control_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torchio as tio
import random
import time
from collections import defaultdict
import traceback

def create_augmentation_transforms():
    """
    Create augmentation transforms that focus on intensity and local transformations.
    Avoids large structural changes like rotation/shift.
    """
    # Intensity-based augmentations (prioritize faster transforms)
    # Order: faster transforms first (Noise, Gamma) before slower ones (Anisotropy, Ghosting)
    intensity_transforms = [
        tio.RandomNoise(std=(0.01, 0.1), p=0.7),  # Fast
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.6),  # Fast
        tio.RandomBiasField(coefficients=0.1, order=3, p=0.5),  # Medium
        tio.RandomSpike(num_spikes=(1, 3), intensity=(0.5, 1.0), p=0.3),  # Medium
        tio.RandomAnisotropy(downsampling=(1.5, 1.5), image_interpolation='linear', p=0.3),  # Slow
        tio.RandomGhosting(intensity=(0.5, 1.0), p=0.3),  # Slow
    ]
    
    # Local transformations
    local_transforms = [
        tio.RandomElasticDeformation(
            num_control_points=(5, 5, 5),  # At least 5 control points required when locked_borders=2
            max_displacement=(4, 4, 4),  # Small displacement to avoid large changes
            locked_borders=2,
            p=0.5
        ),
    ]
    
    # Patch-based augmentations (will be applied separately)
    # These will be applied manually in the processing loop
    
    return intensity_transforms, local_transforms

def visualize_slice(volume, title, save_path, rank=0):
    """
    Visualize a middle slice of a 3D volume.
    
    Args:
        volume: Tensor of shape (1, C, D, H, W) or (C, D, H, W)
        title: Title for the plot
        save_path: Path to save the image
        rank: Process rank (only rank 0 saves)
    """
    if rank != 0:
        return
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    # Handle both (1, C, D, H, W) and (C, D, H, W) formats
    if volume.dim() == 5:
        volume = volume.squeeze(0)  # (C, D, H, W)
    
    # Extract middle slice along D dimension, channel 0
    # volume is (C, D, H, W), take [0, D//2, :, :] -> (H, W)
    slice_data = volume[0, volume.shape[1]//2, :, :].cpu().numpy()
    
    # Normalize to [0, 1] for visualization
    slice_min = slice_data.min()
    slice_max = slice_data.max()
    if slice_max > slice_min:
        slice_data = (slice_data - slice_min) / (slice_max - slice_min)
    else:
        slice_data = np.zeros_like(slice_data)
    
    # Create visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(slice_data, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def apply_patch_augmentations(data, p_blur=0.5, p_delete=0.3):
    """
    Apply random patch-based gaussian blur and deleting.
    
    Args:
        data: Tensor of shape (C, D, H, W)
        p_blur: Probability of applying blur to a patch
        p_delete: Probability of deleting a patch
    """
    C, D, H, W = data.shape
    
    # Random patch size (small patches for local effects)
    patch_size = random.randint(32, 64)
    
    # Random number of patches to modify
    num_patches = random.randint(0, 30)
    
    for _ in range(num_patches):
        # Random patch location
        d_start = random.randint(0, max(1, D - patch_size))
        h_start = random.randint(0, max(1, H - patch_size))
        w_start = random.randint(0, max(1, W - patch_size))
        
        d_end = min(d_start + patch_size, D)
        h_end = min(h_start + patch_size, H)
        w_end = min(w_start + patch_size, W)
        
        if random.random() < p_delete:
            # Delete patch (set to zero or mean)
            data[:, d_start:d_end, h_start:h_end, w_start:w_end] = data[:, 0, 0, 0]
        elif random.random() < p_blur:
            # Apply gaussian blur to patch
            patch = data[:, d_start:d_end, h_start:h_end, w_start:w_end].clone()
            # Simple gaussian blur using convolution
            kernel_size = 3
            sigma = random.uniform(0.5, 1.5)
            # Create gaussian kernel (ensure kernel_size elements)
            coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
            kernel_1d = torch.exp(-coords**2 / (2*sigma**2))
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
            kernel_3d = kernel_3d / kernel_3d.sum()
            kernel_3d = kernel_3d.view(1, 1, kernel_size, kernel_size, kernel_size).to(data.device)
            
            # Apply blur to each channel
            for c in range(C):
                patch_c = patch[c:c+1, :, :, :].unsqueeze(0)  # (1, 1, D, H, W)
                blurred = torch.nn.functional.conv3d(
                    patch_c, kernel_3d, padding=kernel_size//2
                ).squeeze(0).squeeze(0)  # (D, H, W)
                data[c, d_start:d_end, h_start:h_end, w_start:w_end] = blurred
    
    return data

def main(args):
    # Setup DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, device={device}")

    # Create output directory
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    dist.barrier()

    # Load AE
    print(f"Rank {rank}: Loading AE...")
    AE = patchvolumeAE.load_from_checkpoint(args.AE_ckpt).to(device)
    AE.eval()

    # Setup Dataset (without latent_root to load raw images)
    dataset = Control_dataset(args.data_path, resolution=args.resolution, downsample_factor=args.downsample_factor, latent_root=None)
    
    # Create augmentation transforms
    intensity_transforms, local_transforms = create_augmentation_transforms()
    
    # Calculate target image size for transform
    target_image_size = tuple([r * args.downsample_factor for r in args.resolution])
    crop_or_pad_transform = tio.CropOrPad(target_image_size)
    
    total_files = len(dataset)
    indices = list(range(total_files))
    
    # Limit number of samples if specified (for testing/profiling)
    if args.max_samples is not None:
        indices = indices[:args.max_samples]
        if rank == 0:
            print(f"Limiting processing to {args.max_samples} samples (for testing/profiling)")
    
    # Split indices among ranks
    my_indices = indices[rank::dist.get_world_size()]
    
    # Set random seed for reproducibility
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    
    # Profiling statistics
    timings = defaultdict(list)
    profile_interval = 10  # Profile every N samples
    sample_count = 0
    
    for idx in tqdm(my_indices, disable=(rank!=0), desc=f"Rank {rank}"):
        item = list(dataset.all_files[idx].items())[0]
        path = item[1]
        
        original_name = os.path.basename(path)
        stem = original_name.replace('.nii.gz', '').replace('.nii', '')
        
        # Load original image directly from file
        try:
            t0 = time.time()
            # Load raw image using torchio
            img = tio.ScalarImage(path)
            img = crop_or_pad_transform(img)
            data = img.data.to(torch.float32)
            t_load = time.time() - t0
            timings['load_image'].append(t_load)
            
            t0 = time.time()
            # Normalize to [-1, 1]
            d_min = data.min()
            d_max = data.max()
            if d_max > d_min:
                data = (data - d_min) / (d_max - d_min) * 2.0 - 1.0
            else:
                data = torch.zeros_like(data)
            t_norm = time.time() - t0
            timings['normalize'].append(t_norm)
            
            t0 = time.time()
            # Ensure data shape is correct after CropOrPad (fallback resize if needed)
            expected_shape = (1, target_image_size[0], target_image_size[1], target_image_size[2])
            if data.shape != expected_shape:
                from torch.nn.functional import interpolate
                data = data.unsqueeze(0)  # (1, C, W, H, D)
                data = interpolate(data, size=target_image_size, mode='trilinear', align_corners=False)
                data = data.squeeze(0)  # (C, W, H, D)
            
            # Transpose dimensions to match AE expectation (C, D, H, W)
            # torchio gives (C, W, H, D), we need (C, D, H, W)
            # transpose(1, 3) swaps W and D: (C, W, H, D) -> (C, D, H, W)
            data = data.transpose(1, 3)  # (C, D, H, W)
            
            # Move to GPU
            x = data.unsqueeze(0).to(device)  # (1, C, D, H, W)
            t_transfer = time.time() - t0
            timings['transfer_to_gpu'].append(t_transfer)
            
            # First, encode original without augmentation (if needed)
            # But we want augmented versions, so skip original
            
            # Pre-compute normalization values once
            with torch.no_grad():
                min_val = AE.codebook.embeddings.min()
                max_val = AE.codebook.embeddings.max()
            
            # Save original image for visualization if enabled
            if getattr(args, 'save_visualization', False):
                vis_dir = os.path.join(args.output_dir, 'visualizations', stem)
                os.makedirs(vis_dir, exist_ok=True)
                # Use unified visualization function
                visualize_slice(x, f'Original: {stem}', 
                               os.path.join(vis_dir, 'original.jpg'), rank)
            
            # Generate num_augmentations augmented versions
            for aug_idx in range(args.num_augmentations):
                t_aug_total = time.time()
                
                # Clone original data
                # data is (C, D, H, W) on CPU
                x_aug = data.clone()
                
                t0 = time.time()
                # Convert to torchio format for augmentation
                # transpose(1, 3) swaps D and W: (C, D, H, W) -> (C, W, H, D)
                x_aug_tio = x_aug.transpose(1, 3)  # (C, W, H, D)
                t_format_conv1 = time.time() - t0
                timings['format_convert_cpu'].append(t_format_conv1)
                
                t0 = time.time()
                # Create torchio subject
                subject = tio.Subject(image=tio.ScalarImage(tensor=x_aug_tio))
                
                # Apply random intensity transforms (reduced from 1-3 to 0-2 for speed)
                num_intensity = random.randint(0, 2)
                selected_intensity = random.sample(intensity_transforms, min(num_intensity, len(intensity_transforms)))
                for transform in selected_intensity:
                    subject = transform(subject)
                t_torchio_intensity = time.time() - t0
                timings['torchio_intensity'].append(t_torchio_intensity)
                
                t0 = time.time()
                # Apply local transforms (elastic deformation) - reduced probability from 50% to 30%
                if random.random() < 0.3:
                    for transform in local_transforms:
                        subject = transform(subject)
                t_torchio_local = time.time() - t0
                timings['torchio_local'].append(t_torchio_local)
                
                t0 = time.time()
                # Convert back to our format
                x_aug = subject['image'].data  # (C, W, H, D) on CPU
                
                # IMPORTANT: Re-apply CropOrPad to ensure correct dimensions after augmentation
                # Some augmentations (like elastic deformation) may change the spatial dimensions
                # Convert back to torchio format for CropOrPad
                subject_aug = tio.Subject(image=tio.ScalarImage(tensor=x_aug))
                subject_aug = crop_or_pad_transform(subject_aug)  # Re-apply CropOrPad to target_image_size
                x_aug = subject_aug['image'].data  # (C, W, H, D) on CPU
                
                # transpose(1, 3) swaps W and D: (C, W, H, D) -> (C, D, H, W)
                x_aug = x_aug.transpose(1, 3)  # (C, D, H, W)
                
                x_aug = x_aug.unsqueeze(0).to(device)  # (1, C, D, H, W) on GPU
                t_format_conv2 = time.time() - t0
                timings['format_convert_gpu'].append(t_format_conv2)
                
                t0 = time.time()
                # Apply patch-based augmentations
                if random.random() < 0.5:
                    x_aug = apply_patch_augmentations(x_aug.squeeze(0), p_blur=0.5, p_delete=0.3).unsqueeze(0)
                t_patch_aug = time.time() - t0
                timings['patch_augmentations'].append(t_patch_aug)
                
                # Save augmented image for visualization if enabled
                # Use unified visualization function to ensure consistency
                if getattr(args, 'save_visualization', False):
                    vis_dir = os.path.join(args.output_dir, 'visualizations', stem)
                    visualize_slice(x_aug, f'Augmented {aug_idx:02d}: {stem}',
                                   os.path.join(vis_dir, f'aug{aug_idx:02d}.jpg'), rank)
                
                t0 = time.time()
                # Encode to latent
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        x_aug = x_aug.transpose(3, 4)  # (1, C, D, H, W) -> (1, C, D, W, H)
                        # requirement for AE: (1, C, D, W, H) according to Singleres_dataset.py
                        embeddings, _ = AE.encode(x_aug, include_embeddings=True, quantize=True)
                    
                    # Normalize (using pre-computed values)
                    z = (embeddings - min_val) / (max_val - min_val) * 2.0 - 1.0
                t_encode = time.time() - t0
                timings['encode'].append(t_encode)
                
                t0 = time.time()
                # Save augmented latent
                save_path = os.path.join(args.output_dir, f"{stem}_aug{aug_idx:02d}.pt")
                
                # Skip if file already exists (unless force is set)
                if os.path.exists(save_path) and not getattr(args, 'overwrite', False):
                    if rank == 0 and aug_idx == 0:
                        print(f"Skipping {stem} - augmented latents already exist. Use --overwrite to regenerate.")
                    continue
                
                # Use faster save (no compression)
                z_cpu = z.squeeze(0).cpu()
                torch.save(z_cpu, save_path, _use_new_zipfile_serialization=False)
                t_save = time.time() - t0
                timings['save'].append(t_save)
                
                t_aug_total = time.time() - t_aug_total
                timings['aug_total'].append(t_aug_total)
                
                # Clear variables
                del x_aug
                del embeddings
                del z
            
            # Print profiling info periodically
            sample_count += 1
            if rank == 0 and sample_count % profile_interval == 0:
                print(f"\n[Profiling after {sample_count} samples]")
                for step_name, times in timings.items():
                    if len(times) > 0:
                        # Calculate based on samples, not augmentations
                        recent_times = times[-profile_interval*args.num_augmentations:] if step_name in ['format_convert_cpu', 'torchio_intensity', 'torchio_local'] else times[-profile_interval:]
                        if len(recent_times) > 0:
                            avg_time = np.mean(recent_times)
                            total_time = np.sum(recent_times)
                            print(f"  {step_name:25s}: avg={avg_time*1000:6.2f}ms, total={total_time:6.2f}s (n={len(recent_times)})")
                print()
            
            # Clear original
            del x
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            traceback.print_exc()
            # Only clear cache on error to recover
            torch.cuda.empty_cache()
    
    # Print final profiling summary
    if rank == 0:
        print("\n" + "="*60)
        print("FINAL PROFILING SUMMARY")
        print("="*60)
        # Calculate total time first
        total_time_all = sum(np.sum(times) for times in timings.values() if len(times) > 0)
        for step_name, times in sorted(timings.items()):
            if len(times) > 0:
                avg_time = np.mean(times) * 1000  # ms
                total_time = np.sum(times)
                pct = (total_time / total_time_all * 100) if total_time_all > 0 else 0
                print(f"{step_name:25s}: avg={avg_time:8.2f}ms, total={total_time:8.2f}s ({pct:5.1f}%), count={len(times):5d}")
        print(f"{'TOTAL':25s}: {'':8s}  total={total_time_all:8.2f}s")
        print("="*60 + "\n")
    
    # Clear cache at the end of script (since augmentation and training are separate)
    torch.cuda.empty_cache()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--AE-ckpt", type=str, required=True)
    parser.add_argument('--resolution', nargs='+', type=int, default=[48, 48, 48])
    parser.add_argument("--downsample-factor", type=int, default=4)
    parser.add_argument("--num-augmentations", type=int, default=10, help="Number of augmented versions per sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing augmented latents")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to process (for testing/profiling)")
    parser.add_argument("--save-visualization", action='store_true', default=False, 
                        help="Save visualization images (before/after augmentation) as JPG files")
    
    args = parser.parse_args()
    main(args)

