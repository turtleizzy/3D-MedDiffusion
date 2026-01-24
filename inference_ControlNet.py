import sys
import os
import argparse
import torch
import torchio as tio
import numpy as np
from collections import OrderedDict
from ddpm.BiFlowNet import BiFlowNet, GaussianDiffusion
from ddpm.ControlNet import ControlNet, ControlledBiFlowNet
from AutoEncoder.model.PatchVolume import patchvolumeAE

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

def load_and_encode_nifti(nifti_path, AE, target_resolution, downsample_factor, device):
    """
    Load a NIfTI file and encode it to latent space using the AutoEncoder.
    
    Args:
        nifti_path: Path to the input NIfTI file
        AE: AutoEncoder model
        target_resolution: Target latent resolution [D, H, W]
        downsample_factor: AE downsample factor (4 or 8)
        device: Device to use for encoding
        
    Returns:
        latent: Encoded latent tensor (1, C, D, H, W)
    """
    print(f"Loading input NIfTI: {nifti_path}")
    
    # Calculate target image size from latent resolution
    target_image_size = tuple([r * downsample_factor for r in target_resolution])
    
    # Load and preprocess image
    img = tio.ScalarImage(nifti_path)
    transform = tio.CropOrPad(target_image_size)
    img = transform(img)
    data = img.data.to(torch.float32)
    
    # Normalize to [-1, 1]
    d_min = data.min()
    d_max = data.max()
    if d_max > d_min:
        data = (data - d_min) / (d_max - d_min) * 2.0 - 1.0
    else:
        data = torch.zeros_like(data)
    
    # Transpose dimensions to match AE expectation (C, D, H, W)
    # Assuming input is (C, W, H, D) from tio
    data = data.transpose(1, 3).transpose(2, 3)
    # Ensure data is on the same device as AE
    # AE might be on a different device (e.g. cuda:1) than ddpm_device (cuda:0)
    # We should move data to AE's device for encoding
    ae_device = next(AE.parameters()).device
    data = data.unsqueeze(0).to(ae_device)  # Add batch dimension
    
    print(f"Input image shape (after preprocessing): {data.shape}")
    
    # Encode to latent
    with torch.no_grad():
        embeddings, _ = AE.encode(data, include_embeddings=True, quantize=True)
        # Normalize embeddings to [-1, 1]
        min_val = AE.codebook.embeddings.min()
        max_val = AE.codebook.embeddings.max()
        latent = (embeddings - min_val) / (max_val - min_val) * 2.0 - 1.0
    
    # Move latent back to target device (ddpm_device)
    latent = latent.to(device)
    
    print(f"Encoded latent shape: {latent.shape}")
    print(f"Encoded latent range: [{latent.min():.4f}, {latent.max():.4f}]")
    
    return latent

def load_models(args, ddpm_device, decoder_device):
    # Calculate control channels based on whether noisy latent control is used
    # Use noisy latent control if: 1) explicitly enabled, or 2) input is provided
    use_noisy_latent = args.use_noisy_latent_control or args.input_nifti or args.input_latent
    
    control_channels = 2  # age + sex
    if use_noisy_latent:
        control_channels += args.volume_channels
        print(f"Using noisy latent control. Control channels: {control_channels}")
    else:
        print(f"Using age/sex only control. Control channels: {control_channels}")
    
    # 1. Base Model
    print("Loading Base Model...")
    base_model = BiFlowNet(
        dim=args.model_dim,
        dim_mults=args.dim_mults,
        channels=args.volume_channels,
        init_kernel_size=3,
        cond_classes=args.num_classes,
        learn_sigma=False,
        use_sparse_linear_attn=args.use_attn,
        vq_size=args.vq_size,
        num_mid_DiT=args.num_dit,
        patch_size=args.patch_size
    ).to(ddpm_device)

    # Load Base Weights
    if os.path.exists(args.base_ckpt):
        checkpoint = torch.load(args.base_ckpt, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        base_model.load_state_dict(new_state_dict, strict=True)
    else:
        print(f"Warning: Base checkpoint {args.base_ckpt} not found. Using random init.")

    # 2. ControlNet
    print("Loading ControlNet...")
    controlnet = ControlNet(base_model, control_channels=control_channels).to(ddpm_device)
    
    # Load ControlNet Weights (if provided)
    if args.control_ckpt and os.path.exists(args.control_ckpt):
        checkpoint = torch.load(args.control_ckpt, map_location='cpu')
        state_dict = checkpoint['controlnet'] if 'controlnet' in checkpoint else checkpoint
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        controlnet.load_state_dict(new_state_dict, strict=True)
    else:
        print("Warning: ControlNet checkpoint not found or not provided. Using random/init weights.")

    model = ControlledBiFlowNet(base_model, controlnet, scale=args.control_scale).to(ddpm_device)
    model.eval()

    # 3. AutoEncoder
    print("Loading AutoEncoder...")
    if os.path.exists(args.ae_ckpt):
        AE = patchvolumeAE.load_from_checkpoint(args.ae_ckpt).to(decoder_device[0])
        AE.enable_decoder_parallel(device_ids = decoder_device)
        AE.eval()
    else:
        print(f"Error: AE checkpoint {args.ae_ckpt} not found.")
        return None, None, None

    # 4. Diffusion
    diffusion = GaussianDiffusion(
        channels=args.volume_channels,
        timesteps=args.timesteps,
        loss_type='l1',
    ).to(ddpm_device)

    return model, AE, diffusion

def generate(args):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ddpm_device = torch.device('cuda:0')
    decoder_device = list(range(1, torch.cuda.device_count()))
    
    model, AE, diffusion = load_models(args, ddpm_device, decoder_device)
    if model is None:
        return

    # Prepare inputs
    # Class mapping: 3:'MRTIBrain', 4:'MRT2Brain'
    if args.modality == 'T1':
        cls_idx = 3
    elif args.modality == 'T2':
        cls_idx = 4
    else:
        raise ValueError("Modality must be T1 or T2")
    
    print(f"Generating {args.modality} (Class {cls_idx}) with Age={args.age}, Sex={args.sex}...")
    
    y = torch.tensor([cls_idx], device=ddpm_device)
    
    # Resolution/Shape
    # If 8x AE, and output 256 -> latent 32
    # args.resolution is latent resolution
    volume_size = args.resolution
    
    # Random noise for diffusion sampling
    z = torch.randn(1, args.volume_channels, volume_size[0], volume_size[1], volume_size[2], device=ddpm_device)
    
    # Resolution embedding
    res = torch.tensor(volume_size, device=ddpm_device)/64.0
    
    # Load input latent if provided (from NIfTI or pre-computed latent)
    input_latent = None
    if args.input_nifti:
        # Calculate downsample factor from resolution
        # 4x: resolution 48 -> image 192, 8x: resolution 24/32 -> image 192/256
        downsample_factor = args.downsample_factor
        input_latent = load_and_encode_nifti(
            args.input_nifti, AE, volume_size, downsample_factor, ddpm_device
        )
    elif args.input_latent:
        print(f"Loading input latent: {args.input_latent}")
        input_latent = torch.load(args.input_latent).unsqueeze(0).to(ddpm_device)
        print(f"Input latent shape: {input_latent.shape}")
    
    # Build Control Condition
    # Determine if using noisy latent control mode
    use_noisy_latent = args.use_noisy_latent_control or args.input_nifti or args.input_latent
    
    if use_noisy_latent:
        # Create control with age, sex, and noisy latent
        control_channels = 2 + args.volume_channels
        control = torch.zeros((1, control_channels, volume_size[0], volume_size[1], volume_size[2]), device=ddpm_device)
        control[:, 0] = args.age
        control[:, 1] = args.sex
        
        if input_latent is not None:
            # Add gaussian noise to input latent
            noise_strength = args.noise_strength
            gaussian_noise = torch.randn_like(input_latent) * noise_strength
            noisy_latent = input_latent + gaussian_noise
            print(f"Using input latent with noise_strength={noise_strength}")
            control[:, 2:] = noisy_latent
            print(f"Control shape: {control.shape}")
            print(f"Noisy latent range: [{noisy_latent.min():.4f}, {noisy_latent.max():.4f}]")
        else:
            # No input provided, use zeros for latent channels
            # control[:, 2:] is already zeros, no need to modify
            print(f"No input latent provided, using zeros for latent channels")
        
    else:
        # Original behavior: age + sex only
        control = torch.zeros((1, 2, volume_size[0], volume_size[1], volume_size[2]), device=ddpm_device)
        control[:, 0] = args.age
        control[:, 1] = args.sex
        print(f"Control shape: {control.shape} (age/sex only)")
    
    # Sampling
    with torch.no_grad():
        samples = diffusion.p_sample_loop(
            model, z, y=y, res=res, hint=control
        )
        samples = samples.to(f"cuda:{decoder_device[0]}")
        
        # Free up memory before decoding
        del model
        del diffusion
        del control
        del z
        torch.cuda.empty_cache()

        # Decode
        print("Decoding...")
        samples = (((samples + 1.0) / 2.0) * (AE.codebook.embeddings.max() - AE.codebook.embeddings.min())) + AE.codebook.embeddings.min()
        
        # If still OOM, try decoding on CPU if possible (but AE might be on GPU)
        # try:
        volume = AE.decode(samples, quantize=True)
        # except torch.cuda.OutOfMemoryError:
        #     print("OOM during decoding. Moving to CPU...")
        #     torch.cuda.empty_cache()
        #     AE = AE.cpu()
        #     samples = samples.cpu()
        #     volume = AE.decode(samples, quantize=True)
        
        # Save
        volume = volume.detach().squeeze(0).cpu()
        # Rearrange to correct orientation if needed (based on train script)
        volume = volume.transpose(1,3).transpose(1,2)
        
        # Convert from [-1, 1] to [0, 1] for proper visualization
        # AE was trained with input normalized to [-1, 1], so output is also in that range
        volume = (volume + 1.0) / 2.0
        # Clamp to [0, 1] to handle any small overshoots
        volume = torch.clamp(volume, 0, 1)
        
        # Generate output filename
        if args.input_nifti or args.input_latent:
            input_name = os.path.basename(args.input_nifti or args.input_latent).split('.')[0]
            output_filename = f"output_{args.modality}_age{args.age}_sex{args.sex}_noise{args.noise_strength}_{input_name}.nii.gz"
        else:
            output_filename = f"output_{args.modality}_age{args.age}_sex{args.sex}.nii.gz"
        save_path = os.path.join(args.output_dir, output_filename)
        os.makedirs(args.output_dir, exist_ok=True)
        
        tio.ScalarImage(tensor=volume).save(save_path)
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", type=str, default="checkpoints/BiFlowNet_0453500.pt")
    parser.add_argument("--control-ckpt", type=str, default=None, help="Path to trained ControlNet checkpoint")
    parser.add_argument("--ae-ckpt", type=str, default="checkpoints/PatchVolume_8x_s2.ckpt")
    parser.add_argument("--output-dir", type=str, default="results/inference")
    
    # Generation params
    parser.add_argument("--modality", type=str, choices=['T1', 'T2'], default='T1')
    parser.add_argument("--age", type=float, default=0.5, help="Normalized Age (0-1)")
    parser.add_argument("--sex", type=float, default=0.0, help="Sex (0 or 1)")
    parser.add_argument("--control-scale", type=float, default=1.0)
    
    # Noisy latent control params
    parser.add_argument("--use-noisy-latent-control", action='store_true',
                        help="Use noisy latent control mode (required if checkpoint was trained with this)")
    parser.add_argument("--input-nifti", type=str, default=None, 
                        help="Optional input NIfTI file to encode as noisy latent control")
    parser.add_argument("--input-latent", type=str, default=None,
                        help="Optional pre-computed latent file (.pt) as noisy latent control")
    parser.add_argument("--noise-strength", type=float, default=0.5,
                        help="Gaussian noise strength to add to input latent (default: 0.5)")
    parser.add_argument("--downsample-factor", type=int, default=4,
                        help="AE downsample factor (4 or 8), used when encoding input NIfTI")
    
    # Model config (should match training)
    parser.add_argument("--model-dim", type=int, default=72)
    parser.add_argument("--dim-mults", nargs='+', type=int, default=[1,1,2,4,8])
    parser.add_argument("--volume-channels", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--use-attn", nargs='+', type=int, default=[0,0,0,1,1])
    parser.add_argument("--vq-size", type=int, default=64)
    parser.add_argument("--num-dit", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument('--resolution', nargs='+', type=int, default=[32, 32, 32])
    
    args = parser.parse_args()
    generate(args)

