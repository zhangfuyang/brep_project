import torch.nn as nn
from diffusers import UNet3DConditionModel

class DiffusionLatent(nn.Module):
    def __init__(self, config):
        super(DiffusionLatent, self).__init__()
        self.unet = UNet3DConditionModel(
            in_channels=config['n_channels'],
            out_channels=config['n_channels'],
            down_block_types=("CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"),
            up_block_types=("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
            block_out_channels=(320, 640, 1280),
            layers_per_block=2,
            downsample_padding=1,
        )
    
    def forward(self, x, timesteps, face_num):
        # x: bs, 1+M, C, N, N, N
        # timesteps: bs
        # face_num: bs
        t_emb = self.unet.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.unet.dtype)
        emb = self.unet.time_embedding(t_emb, None)

        
        print()

