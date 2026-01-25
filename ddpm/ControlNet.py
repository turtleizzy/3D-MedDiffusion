import torch
from torch import nn
from ddpm.BiFlowNet import BiFlowNet, DiTBlock, FinalLayer, Mlp, PatchEmbed_Voxel, ResnetBlock, AttentionBlock, Downsample, PreNorm, Residual, Upsample, Block
import copy
from einops import rearrange

def exists(x):
    return x is not None

class ZeroConv3d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 1, padding=0)
        nn.init.constant_(self.conv.weight, 0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)

class ControlNet(nn.Module):
    def __init__(self, biflownet_model, control_channels=2):
        super().__init__()
        # We don't store biflownet_model as a member to avoid saving it twice when saving ControlNet
        # But we use it for init.
        
        # Copy encoder parts
        self.init_conv = copy.deepcopy(biflownet_model.init_conv)
        self.channels = biflownet_model.channels
        self.dim = biflownet_model.dim
        self.sub_volume_size = biflownet_model.sub_volume_size
        self.patch_size = biflownet_model.patch_size
        self.vq_size = biflownet_model.vq_size
        self.control_channels = control_channels
        
        # Process control to match init_conv output dimension (zero-initialized)
        self.control_proj = nn.Conv3d(control_channels, self.dim, 
                                     kernel_size=self.init_conv.kernel_size,
                                     padding=self.init_conv.padding)
        nn.init.zeros_(self.control_proj.weight)
        nn.init.zeros_(self.control_proj.bias)

        self.time_mlp = copy.deepcopy(biflownet_model.time_mlp)
        
        self.cond_classes = biflownet_model.cond_classes
        if self.cond_classes is not None:
            self.cond_emb = copy.deepcopy(biflownet_model.cond_emb)
            
        self.res_condition = biflownet_model.res_condition
        if self.res_condition:
            self.res_mlp = copy.deepcopy(biflownet_model.res_mlp)

        self.x_embedder = copy.deepcopy(biflownet_model.x_embedder)
        self.pos_embed = copy.deepcopy(biflownet_model.pos_embed)
        
        self.IntraPatchFlow_input = copy.deepcopy(biflownet_model.IntraPatchFlow_input)
        
        self.downs = copy.deepcopy(biflownet_model.downs)
        self.feature_fusion = biflownet_model.feature_fusion
        
        self.mid_block1 = copy.deepcopy(biflownet_model.mid_block1)
        self.mid_spatial_attn = copy.deepcopy(biflownet_model.mid_spatial_attn)
        self.mid_block2 = copy.deepcopy(biflownet_model.mid_block2)

        self.zero_convs_downs = nn.ModuleList()
        # Track dimensions for zero convs
        dims = [self.dim] + [self.dim * m for m in biflownet_model.dim_mults if m is not None] # Check dim_mults usage in BiFlowNet
        # Actually BiFlowNet: dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # We assume init_dim == dim
        
        # Let's dynamically create zero convs based on what we see in downs
        # But we need to know the channels.
        # We can inspect the layers in downs.
        
        for blocks in self.downs:
            # blocks is a ModuleList: [block1, attn1, block2, attn2, downsample]
            # block1 output channels?
            # block1 is ResnetBlock.
            # We can check block1.res_conv.out_channels if it exists, or infer.
            # Actually, let's look at BiFlowNet constructor again.
            # block1(dim_in, dim_out)
            # block2(dim_out, dim_out)
            # So output is dim_out.
            # We need 2 zero convs per level (after block1 and block2).
            # The 'dim_out' is encoded in the block.
            
            # Let's inspect block1
            # block1 is ResnetBlock. It has block2 (Block) which has norm (GroupNorm).
            # The num_channels of norm is dim_out.
            dim_out = blocks[0].block2.norm.num_channels
            self.zero_convs_downs.append(ZeroConv3d(dim_out))
            self.zero_convs_downs.append(ZeroConv3d(dim_out))
            
        # Mid block output
        mid_dim = self.mid_block2.block2.norm.num_channels
        self.zero_convs_mid = ZeroConv3d(mid_dim)

    def unpatchify_voxels(self, x0):
        c = self.dim
        p = self.patch_size
        x,y,z = torch.tensor(self.sub_volume_size) // self.patch_size
        # assert x * y * z == x0.shape[1]

        x0 = x0.reshape(shape=(x0.shape[0], x, y, z, p, p, p, c))
        x0 = torch.einsum('nxyzpqrc->ncxpyqzr', x0)
        volume = x0.reshape(shape=(x0.shape[0], c, x * p, y * p, z * p))
        return volume

    def forward(self, x, control, time, y=None, res=None):
        b = x.shape[0]
        ori_shape = (x.shape[2]*8,x.shape[3]*8,x.shape[4]*8) # assuming 8x AE
        
        if x.shape[2:] != control.shape[2:]:
             # Resize control to match x
             control = torch.nn.functional.interpolate(control, size=x.shape[2:], mode='nearest')
        
        # Prepare x_IntraPatch for DiT path (needed for h_Unet generation)
        # Must be created BEFORE init_conv, as x_embedder expects channels (not dim) input
        x_IntraPatch = x.clone()
        p = self.sub_volume_size[0]
        x_IntraPatch = x_IntraPatch.unfold(2,p,p).unfold(3,p,p).unfold(4,p,p)
        p1 , p2 , p3= x_IntraPatch.size(2) , x_IntraPatch.size(3) , x_IntraPatch.size(4)
        x_IntraPatch = rearrange(x_IntraPatch , 'b c p1 p2 p3 d h w -> (b p1 p2 p3) c d h w')
        
        # Process control through zero conv and add to x (U-Net path)
        control_proj = self.control_proj(control)
        x = self.init_conv(x) + control_proj

        t = self.time_mlp(time) if exists(self.time_mlp) else None
        c = t.shape[-1]
        t_DiT = t.unsqueeze(1).repeat(1,p1*p2*p3,1).view(-1,c)

        if self.cond_classes:
            assert y.shape == (x.shape[0],)
            cond_emb = self.cond_emb(y)
            cond_emb_DiT = cond_emb.unsqueeze(1).repeat(1,p1*p2*p3,1).view(-1,c)
            t = t + cond_emb
            t_DiT = t_DiT + cond_emb_DiT
        if self.res_condition:
            if len(res.shape) == 1:
                res = res.unsqueeze(0)
            res_condition_emb = self.res_mlp(res)
            t = torch.cat((t,res_condition_emb),dim=1)
            res_condition_emb_DiT = res_condition_emb.unsqueeze(1).repeat(1,p1*p2*p3,1).view(-1,c)
            t_DiT = torch.cat((t_DiT,res_condition_emb_DiT),dim=1)
            
        # Process DiT path (without control injection) to generate h_Unet for U-Net path
        x_IntraPatch = self.x_embedder(x_IntraPatch)
        x_IntraPatch = x_IntraPatch + self.pos_embed
        
        h_Unet = []  # Needed for U-Net path feature fusion
        
        for i, (Block, MlpLayer) in enumerate(self.IntraPatchFlow_input):
            x_IntraPatch = Block(x_IntraPatch,t_DiT)
            
            # Generate Unet_feature for U-Net path fusion
            Unet_feature = self.unpatchify_voxels(MlpLayer(x_IntraPatch,t_DiT))
            Unet_feature = rearrange(Unet_feature, '(b p) c d h w -> b p c d h w', b=b) 
            Unet_feature = rearrange(Unet_feature, 'b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)',
                        p1=ori_shape[0]//self.vq_size, p2=ori_shape[1]//self.vq_size, p3=ori_shape[2]//self.vq_size)
            h_Unet.append(Unet_feature)

        h_ctrl = []
        zero_idx = 0
        
        for idx, (block1, spatial_attn1, block2, spatial_attn2,downsample) in enumerate(self.downs):
            if idx < self.feature_fusion :
                x = x + h_Unet.pop(0)
            
            x = block1(x, t)
            x = spatial_attn1(x)
            # Collect
            h_ctrl.append(self.zero_convs_downs[zero_idx](x))
            zero_idx += 1
            
            x = block2(x, t)
            x = spatial_attn2(x)
            # Collect
            h_ctrl.append(self.zero_convs_downs[zero_idx](x))
            zero_idx += 1
            
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_block2(x, t)
        
        mid_ctrl = self.zero_convs_mid(x)
        
        return {
            'h': h_ctrl,
            'mid': mid_ctrl
        }

class ControlledBiFlowNet(nn.Module):
    def __init__(self, biflownet, controlnet, scale=1.0):
        super().__init__()
        self.biflownet = biflownet
        self.controlnet = controlnet
        self.scale = scale
        
    def forward(self, x, t, y=None, res=None, hint=None, control=None):
        if control is None:
            control = hint
        
        control_states = self.controlnet(x, control, t, y, res)
        # Apply scale if needed? Usually implicit in ZeroConv logic (starts at 0).
        # But global scale is also useful.
        if self.scale != 1.0:
            for k in control_states:
                if isinstance(control_states[k], list):
                    control_states[k] = [v * self.scale for v in control_states[k]]
                else:
                    control_states[k] = control_states[k] * self.scale
                    
        return self.biflownet(x, t, y=y, res=res, control_states=control_states)
