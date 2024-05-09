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
    
    