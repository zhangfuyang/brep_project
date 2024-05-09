import torch.nn as nn
import torch
from model_utils import Timesteps, TimestepEmbedding, \
        get_parameter_dtype

class JointModel(nn.Module):
    def __init__(self, config, ):
        super().__init__()

        timestep_input_dim = 512
        time_embed_dim = 1024
        self.time_proj = Timesteps(num_channels=timestep_input_dim, 
                                   flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim=time_embed_dim)

        self.face_model = LatentModel(config)
        self.solid_model = LatentModel(config)

        # cross-attn
        # TODO
    
    def forward(self, face_latent, solid_latent, timestep):
        # face_latent: (B, M, 512)
        # solid_latent: (B, 1, 512)
        timestep = timestep.to(face_latent.device)
        t_emb = self.time_embedding(self.time_proj(timestep))
        t_emb = t_emb.unsqueeze(1)


        face_latent = self.face_model(face_latent, t_emb)
        solid_latent = self.solid_model(solid_latent, t_emb)
        return face_latent, solid_latent


class LatentModel(nn.Module):
    def __init__(self, config,):
        super().__init__()

        self.residual = True

        # encode
        channels = [1024, 1024, 2048, 2048]
        num_layers = 1
        in_channel = 512
        time_embed_dim = 1024

        self.layers = nn.ModuleList()
        for i in range(len(channels)):
            if i == 0:
                in_ = in_channel
            else:
                in_ = channels[i-1]
            out_ = channels[i]
            self.layers.append(EncodeBlock1D(in_, out_, time_embed_dim, num_layers))
        
        # decode
        out_channel = 512
        self.fc_out = nn.Linear(channels[-1], out_channel)

    def forward(self, x, t_emb):
        # x: (B, M, 512)
        # t_emb: (B, 1, 1024)
        x_ori = x
        for layer in self.layers:
            x = layer(x, t_emb)
        x = self.fc_out(x)
        if self.residual:
            x = x + x_ori
    
        return x

    @property
    def dtype(self) -> torch.dtype:
        return get_parameter_dtype(self)


class EncodeBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, 
                 num_layers):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i < num_layers - 1:
                in_, out_ = in_channels, in_channels
            else:
                in_, out_ = in_channels, out_channels
            self.layers.append(FCTimeEmbed(in_, out_, temb_channels))

    def forward(self, sample, t_emb):
        for layer in self.layers:
            sample = layer(sample, t_emb)
        return sample
    

class FCTimeEmbed(nn.Module):
    def __init__(self, in_channel, out_channel, time_embed_dim):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(in_channel)
        self.fc1 = nn.Linear(in_channel, out_channel)

        self.time_emb_proj = nn.Linear(time_embed_dim, out_channel)

        self.norm2 = torch.nn.LayerNorm(out_channel)
        self.fc2 = nn.Linear(out_channel, out_channel)

        self.nonlinearity = nn.SiLU()

    def forward(self, x, t_emb):
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.fc1(x)

        t_emb = self.time_emb_proj(t_emb)
        x = x + t_emb

        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.fc2(x)
        return x

