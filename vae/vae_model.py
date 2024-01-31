"""Adapted from https://github.com/milesial/Pytorch-UNet/tree/master/unet"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE3D(nn.Module):
    def __init__(self, n_channels, trilinear=True):
        """A simple 3D Unet, adapted from a 2D Unet from https://github.com/milesial/Pytorch-UNet/tree/master/unet
        Arguments:
          n_channels = number of input channels; 3 for RGB, 1 for grayscale input
          trilinear = use trilinear interpolation to upsample; if false, 3D convtranspose layers will be used instead
        """
        super(VAE3D, self).__init__()
        _channels = (32, 64, 128, 256, 512, 512)
        self.latent_dim = 1024
        self.n_channels = n_channels
        self.channels = [int(c) for c in _channels]
        self.trilinear = trilinear
        self.convtype = nn.Conv3d

        self.encoder = nn.Sequential(
            DoubleConv(n_channels, self.channels[0], conv_type=self.convtype),
            Down(self.channels[0], self.channels[1], conv_type=self.convtype),
            Down(self.channels[1], self.channels[2], conv_type=self.convtype),
            Down(self.channels[2], self.channels[3], conv_type=self.convtype),
            Down(self.channels[3], self.channels[4], conv_type=self.convtype),
            Down(self.channels[4], self.channels[5], conv_type=self.convtype),
        )

        self.fc_mu = nn.Linear(self.channels[5]*8, self.latent_dim)
        self.fc_logvar = nn.Linear(self.channels[5]*8, self.latent_dim)

        self.decoder_input = nn.Linear(self.latent_dim, self.channels[5]*8)
        self.decoder = nn.Sequential(
            Up(self.channels[5], self.channels[4], trilinear),
            Up(self.channels[4], self.channels[3], trilinear),
            Up(self.channels[3], self.channels[2], trilinear),
            Up(self.channels[2], self.channels[1], trilinear),
            Up(self.channels[1], self.channels[0], trilinear)
        )
        self.final_layer = nn.Sequential(
            nn.Conv3d(self.channels[0], n_channels, kernel_size=1),
            nn.Tanh()
        )

    def encode(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        logvar = self.fc_logvar(result)

        return mu, logvar
    
    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.channels[4], 2, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar, **kwargs):
        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recon_x, x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss

        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
    
    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)
    

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d, mid_channels=None,
                 no_end_act=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if no_end_act:
            self.double_conv = nn.Sequential(
                conv_type(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(mid_channels),
                nn.LeakyReLU(negative_slope=0.1),
                conv_type(mid_channels, out_channels, kernel_size=3, padding=1),
            )
        else:
            self.double_conv = nn.Sequential(
                conv_type(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(mid_channels),
                nn.LeakyReLU(negative_slope=0.1),
                conv_type(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(negative_slope=0.1),
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            ResNetBlock(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = ResNetBlock(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = ResNetBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()

        if in_channels != out_channels:
            # conv1x1 for increasing the number of channels
            self.conv1 = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.conv1 = nn.Identity()

        # residual block
        self.conv2 = DoubleConv(out_channels, out_channels, nn.Conv3d)
        self.conv3 = DoubleConv(out_channels, out_channels, nn.Conv3d)

        # create non-linearity separately
        self.non_linearity = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        # apply first convolution to bring the number of channels to out_channels
        residual = self.conv1(x)

        # residual block
        out = self.conv2(residual)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out

