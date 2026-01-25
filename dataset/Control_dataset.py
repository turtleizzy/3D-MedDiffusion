from dataset.Singleres_dataset import Singleres_dataset
import torch
import numpy as np
import json
import os
import glob
import torchio as tio


class LatentLoadError(Exception):
    """Custom exception for latent loading errors."""
    pass

class Control_dataset(Singleres_dataset):
    def __init__(self, root_dir=None, resolution=[32,32,32], generate_latents=False, 
                 downsample_factor=8, latent_root=None, volume_channels=8,
                 use_noisy_latent_control=False, max_noise_strength=1.0,
                 full_noise_prob=0.0):
        """
        Args:
            root_dir: Path to index.json or directory containing it
            resolution: Latent resolution (e.g., [24,24,24] for 8x, [48,48,48] for 4x)
            generate_latents: Whether to generate latents from raw images
            downsample_factor: AE downsample factor (8 or 4)
            latent_root: Path to pre-computed latents (for 4x training)
            volume_channels: Number of latent channels (default 8)
            use_noisy_latent_control: Whether to include noisy latent as control
            max_noise_strength: Maximum noise strength for random noise injection
            full_noise_prob: Probability of using zeros instead of noisy latent (0.0-1.0)
        """
        # super().__init__(root_dir=None, resolution=resolution, generate_latents=generate_latents)
        
        self.resolution = resolution
        self.generate_latents = generate_latents
        self.metadata = {}
        self.all_files = []
        self.downsample_factor = downsample_factor
        self.latent_root = latent_root
        self.volume_channels = volume_channels
        self.use_noisy_latent_control = use_noisy_latent_control
        self.max_noise_strength = max_noise_strength
        self.full_noise_prob = full_noise_prob
        
        # Calculate target image size
        self.target_image_size = tuple([r * self.downsample_factor for r in resolution])
        self.transform = tio.CropOrPad(self.target_image_size)

        # Load index.json
        if root_dir.endswith('.json'):
            json_path = root_dir
            data_dir = os.path.dirname(json_path)
        else:
            json_path = os.path.join(root_dir, 'index.json')
            data_dir = root_dir
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        print(f"Loading data from {json_path}...")
        print(f"Control settings: use_noisy_latent_control={use_noisy_latent_control}, max_noise_strength={max_noise_strength}, full_noise_prob={full_noise_prob}")
        
        for entry in data:
            study_uid = entry.get('studyUID')
            age = float(entry.get('age', 0))
            sex_str = entry.get('sex')
            
            # Map sex
            if sex_str == '男':
                sex = 1.0
            elif sex_str == '女':
                sex = 0.0
            else:
                sex = 0.0 
                
            # Find file
            search_pattern = os.path.join(data_dir, f"{study_uid}*_mni.nii.gz")
            found_files = glob.glob(search_pattern)
            
            if found_files:
                file_path = found_files[0]
                # Use class 3 (T1) as default
                self.all_files.append({'3': file_path})
                self.metadata[file_path] = (age, sex)
                
        self.file_num = len(self.all_files)
        print(f"Total files found: {self.file_num}")
        
        # Calculate control channels
        # Base: 2 (age, sex)
        # With noisy latent: 2 + volume_channels
        self.control_channels = 2 + (self.volume_channels if self.use_noisy_latent_control else 0)
        print(f"Control channels: {self.control_channels} (base=2, noisy_latent={self.volume_channels if self.use_noisy_latent_control else 0})")

    def __getitem__(self, index):
        # latent: (C, D, H, W)
        # y: class index
        # res: resolution
        
        item = list(self.all_files[index].items())[0]
        cls_idx = int(item[0])
        path = item[1]
        
        age, sex = self.metadata[path]
        
        # Normalize age 0-1 (assuming max age 100)
        age = age / 100.0
        
        D, H, W = self.resolution
        original_name = os.path.basename(path)
        stem = original_name.replace('.nii.gz', '').replace('.nii', '')

        if self.latent_root is not None:
            # Load pre-computed latent

            # First try to find original latent
            latent_path = os.path.join(self.latent_root, f"{stem}.pt")
            
            if not os.path.exists(latent_path):
                raise LatentLoadError(f"Latent file not found: {latent_path}")
            
            # Load original latent with error handling
            try:
                data_original = torch.load(latent_path, map_location='cpu')
            except Exception as e:
                raise LatentLoadError(f"Failed to load latent file {latent_path}: {str(e)}")
            
            # Validate original latent shape
            expected_shape = (self.volume_channels, D, H, W)
            if data_original.shape != expected_shape:
                raise LatentLoadError(
                    f"Original latent shape mismatch for {latent_path}: "
                    f"expected {expected_shape}, got {data_original.shape}"
                )
            
            # Look for augmented latents: {stem}_aug*.pt
            aug_pattern = os.path.join(self.latent_root, f"{stem}_aug*.pt")
            aug_files = glob.glob(aug_pattern)
            all_options = [latent_path] + aug_files
            
            if len(all_options) == 0:
                raise LatentLoadError(f"No latent files found for {stem}")
            
            # Randomly select one augmented version
            augmented_latent_path = np.random.choice(all_options)
            
            # Load augmented latent with error handling
            try:
                data = torch.load(augmented_latent_path, map_location='cpu')
            except Exception as e:
                raise LatentLoadError(
                    f"Failed to load augmented latent file {augmented_latent_path}: {str(e)}"
                )
            
            # Validate augmented latent shape
            if data.shape != expected_shape:
                raise LatentLoadError(
                    f"Augmented latent shape mismatch for {augmented_latent_path}: "
                    f"expected {expected_shape}, got {data.shape}"
                )
            
            # Ensure data is float32
            if data.dtype != torch.float32:
                data = data.to(torch.float32)
            if data_original.dtype != torch.float32:
                data_original = data_original.to(torch.float32)

            # Create control tensor
            control = self._create_control(age, sex, data)
            
            return data_original, torch.tensor(cls_idx), torch.tensor(self.resolution)/64.0, control
        else:
            # Load raw image
            img = tio.ScalarImage(path)
            img = self.transform(img)
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
            data = data.transpose(1,3).transpose(2,3)
            
            # For raw images, we can't create noisy latent control without AE
            # Create basic control with age/sex only
            control = self._create_control(age, sex, None)
            
            return data, torch.tensor(cls_idx), torch.tensor(self.resolution)/64.0, control
    
    def _create_control(self, age, sex, latent=None):
        """
        Create control tensor with age, sex, and optionally noisy latent.
        
        Args:
            age: Normalized age (0-1)
            sex: Sex indicator (0 or 1)
            latent: Optional latent tensor (C, D, H, W) for noisy latent control
            
        Returns:
            control: Tensor of shape (control_channels, D, H, W)
        """
        D, H, W = self.resolution
        
        if self.use_noisy_latent_control:
            # Create control with age, sex, and noisy latent (or zeros if no latent)
            # Shape: (2 + volume_channels, D, H, W)
            control = torch.zeros((self.control_channels, D, H, W), dtype=torch.float32)
            
            # Fill age and sex channels
            control[0] = age
            control[1] = sex
            
            if latent is not None:
                # Validate latent shape
                expected_latent_shape = (self.volume_channels, D, H, W)
                if latent.shape != expected_latent_shape:
                    raise LatentLoadError(
                        f"Latent shape mismatch in _create_control: "
                        f"expected {expected_latent_shape}, got {latent.shape}"
                    )
                
                # With full_noise_prob probability, use zeros instead of noisy latent
                if torch.rand(1).item() < self.full_noise_prob:
                    # Use zeros for latent channels
                    # control[2:] is already zeros, no need to modify
                    pass
                else:
                    # Randomly sample noise strength for this sample
                    noise_strength = torch.rand(1).item() * self.max_noise_strength
                    
                    # Create noisy latent: latent + noise_strength * gaussian_noise
                    gaussian_noise = torch.randn_like(latent)
                    noisy_latent = latent + noise_strength * gaussian_noise
                    
                    # Fill noisy latent channels
                    control[2:] = noisy_latent
            else:
                # No latent provided, use zeros for latent channels
                # control[2:] is already zeros, no need to modify
                pass
            
        else:
            # Original behavior: only age and sex
            control = torch.zeros((2, D, H, W), dtype=torch.float32)
            control[0] = age
            control[1] = sex
            
        return control
    
    def get_control_channels(self):
        """Return the number of control channels."""
        return self.control_channels

