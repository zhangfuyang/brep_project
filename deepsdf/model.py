import torch
import torch.nn as nn
import torch.nn.functional as F

class SDFModel(nn.Module):
    def __init__(self, latent_code_size=512,
                 channels=(2, 4, 8, 8), **kwargs):
        super(SDFModel, self).__init__()

        self.latent_code_size = latent_code_size
        self.margin_beta = kwargs['margin_beta']

        self.model = nn.ModuleList()
        in_channels = self.latent_code_size + 3
        for i in range(len(channels)):
            out_channels = channels[i]
            self.model.append(
                nn.Sequential(
                    nn.Linear(in_channels, out_channels),
                    nn.LayerNorm(out_channels),
                    nn.SiLU()
                )
            )
            in_channels = out_channels
        
        self.model.append(
            nn.Sequential(
                nn.Linear(in_channels, 1),
                nn.Sigmoid()
            )
        )

    def forward(self, point, latent_code):
        # point: N x 3
        # latent_code: N x latent_code_size
        x = torch.cat([point, latent_code], dim=-1)
        for model in self.model:
            x = model(x)
        x = x * (self.margin_beta[1] - self.margin_beta[0]) + self.margin_beta[0]
        return x
    
    
class Mid(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=2):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(ResNetBlock(in_channels, out_channels))
            in_channels = out_channels
        
    def forward(self, x):
        for model in self.res_blocks:
            x = model(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=2, with_conv=True):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(ResNetBlock(in_channels, out_channels))
            in_channels = out_channels
        
        self.with_conv = with_conv
        if with_conv:
            self.downsample = nn.Conv3d(in_channels, out_channels, 
                                    kernel_size=3, stride=2, padding=0)
        else:
            self.downsample = nn.AvgPool3d(2)

    def forward(self, x):
        for model in self.res_blocks:
            x = model(x)
        if self.with_conv:
            pad = (0,1,0,1,0,1)
            x = F.pad(x, pad, "constant", 0)
        return self.downsample(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=2, with_conv=True):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(ResNetBlock(in_channels, out_channels))
            in_channels = out_channels
        
        self.with_conv = with_conv
        if with_conv:
            self.upsample = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x):
        for model in self.res_blocks:
            x = model(x)
        return self.upsample(x)

def Normalize(in_channels, num_groups=32):
    #return torch.nn.BatchNorm3d(num_features=in_channels)
    return torch.nn.InstanceNorm3d(num_features=in_channels)
    #return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = Normalize(out_channels)
        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                 out_channels,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1)
        
        self.nonlinearity = nn.SiLU()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x+h

