import torch.nn as nn
import torch
from model_utils import Timesteps, TimestepEmbedding, \
        get_parameter_dtype, EncodeBlock3D, EncodeAttnBlock3D, Down

class Solid3DModel(nn.Module):
    def __init__(self,
                in_channels = 8,
                out_channels = 8,
                freq_shift = 0,
                flip_sin_to_cos = True,
                voxel_block_types = ("EncodeBlock3D", "EncodeAttnBlock3D", "EncodeAttnBlock3D"),
                face_block_types = ("EncodeBlock3D", "EncodeAttnBlock3D", "EncodeAttnBlock3D"),
                block_channels = (224, 448, 672),
                layers_per_block = 2,
                act_fn: str = "silu",
                attention_head_dim = 8,
                norm_num_groups = 32,
                norm_eps: float = 1e-5,
                cross_attn_zero_init = True,
                cross_down_sample = False,):
        super().__init__()
        self.cond_solid_unet = UNet3DModel(in_channels, out_channels, freq_shift, 
                                      flip_sin_to_cos, voxel_block_types, 
                                      block_channels, 1, 
                                      act_fn, attention_head_dim, norm_num_groups, 
                                      norm_eps, is_cond=True, no_time_embed=True)
        self.cond_face_unet = UNet3DModel(in_channels, out_channels, freq_shift,
                                        flip_sin_to_cos, voxel_block_types, block_channels, 
                                        1, act_fn, attention_head_dim, 
                                        norm_num_groups, norm_eps, is_cond=True, no_time_embed=True)
        self.face_unet = UNet3DModel(in_channels, out_channels, freq_shift, 
                                     flip_sin_to_cos, face_block_types, block_channels, 
                                     layers_per_block, act_fn, attention_head_dim, 
                                     norm_num_groups, norm_eps)

        self.f2f_attn = nn.ModuleList()
        for i in range(len(block_channels)):
            self.f2f_attn.append(
                EncodeAttnBlock3D(
                    in_channels=block_channels[i],
                    out_channels=block_channels[i],
                    temb_channels=self.face_unet.time_embed_dim,
                    num_layers=layers_per_block,
                    resnet_act_fn=act_fn,
                    attention_head_dim=attention_head_dim,
                    num_groups=norm_num_groups,
                    eps=norm_eps,
                    zero_init=cross_attn_zero_init,
                    attention_residual=False
                )
            )
        if cross_down_sample:
            self.f2f_down = nn.ModuleList()
            for i in range(len(block_channels)):
                self.f2f_down.append(
                    nn.Sequential(
                        Down(block_channels[i], block_channels[i], num_res_blocks=1, with_conv=False),
                        Down(block_channels[i], block_channels[i], num_res_blocks=1, with_conv=False)
                    )

                )
        else:
            self.f2f_down = None
        self.s2f_attn = nn.ModuleList()
        for i in range(len(block_channels)):
            self.s2f_attn.append(
                EncodeAttnBlock3D(
                    in_channels=block_channels[i],
                    out_channels=block_channels[i],
                    temb_channels=self.face_unet.time_embed_dim,
                    num_layers=layers_per_block,
                    resnet_act_fn=act_fn,
                    attention_head_dim=attention_head_dim,
                    num_groups=norm_num_groups,
                    eps=norm_eps,
                    zero_init=cross_attn_zero_init,
                    attention_residual=False
                )
            )
        if cross_down_sample:
            self.s2f_down = nn.ModuleList()
            for i in range(len(block_channels)):
                self.s2f_down.append(
                    nn.Sequential(
                        Down(block_channels[i], block_channels[i], num_res_blocks=layers_per_block, with_conv=False),
                        Down(block_channels[i], block_channels[i], num_res_blocks=layers_per_block, with_conv=False)
                    )
                )
        else:
            self.s2f_down = None
    
    
    def forward(self, x, face_cond, solid_cond, timestep):

        # x: bs, ch, n, n, n
        # face_cond: bs, m, ch, n, n, n
        # solid_cond: bs, 1, ch, n, n, n
        
        face_t_emb = self.face_unet.time_encode(timestep, x.shape[0], x.device)

        cond_solid_latent = self.cond_solid_unet.conv_in_model(solid_cond) # bs, 1, ch, n, n, n
        cond_face_latent = self.cond_face_unet.conv_in_model(face_cond) # bs, m, ch, n, n, n
        face_latent = self.face_unet.conv_in_model(x) # bs, ch, n, n, n

        for encode_block_i in range(len(self.cond_solid_unet.encode_blocks)):
            cond_solid_latent, _ = self.cond_solid_unet.encode_block_n(cond_solid_latent, None, encode_block_i)
            cond_face_latent, _ = self.cond_face_unet.encode_block_n(cond_face_latent, None, encode_block_i)
            face_latent, _ = self.face_unet.encode_block_n(face_latent, face_t_emb, encode_block_i)
            # cond_solid_latent: bs, 1, ch, n, n, n
            # cond_face_latent: bs, m, ch, n, n, n
            # face_latent: bs, ch, n, n, n

            if self.s2f_down is not None:
                bs = cond_solid_latent.shape[0]
                m = cond_solid_latent.shape[1]
                cond_solid_cross_latent = self.s2f_down[encode_block_i](cond_solid_latent.reshape(bs * m, *cond_solid_latent.shape[2:]))
                cond_solid_cross_latent = cond_solid_cross_latent.reshape(bs, m, *cond_solid_cross_latent.shape[1:])
            if self.f2f_down is not None:
                bs = cond_face_latent.shape[0]
                m = cond_face_latent.shape[1]
                cond_face_cross_latent = self.f2f_down[encode_block_i](cond_face_latent.reshape(bs * m, *cond_face_latent.shape[2:]))
                cond_face_cross_latent = cond_face_cross_latent.reshape(bs, m, *cond_face_cross_latent.shape[1:])

            # extrct cross_attn latent for face_latent
            bs = face_latent.shape[0]
            
            # 1. f2f
            face_latent = face_latent + self.f2f_attn[encode_block_i](
                face_latent, face_t_emb, 
                encoder_hidden_states=cond_face_cross_latent)[0]
            # 2. v2f
            face_latent = face_latent + self.s2f_attn[encode_block_i](
                face_latent, face_t_emb, 
                encoder_hidden_states=cond_solid_cross_latent)[0]
            
        face_latent = self.face_unet.conv_out_model(face_latent)

        x = x + face_latent

        return x

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)


class UNet3DModel(nn.Module):
    def __init__(
        self,
        in_channels = 8,
        out_channels = 8,
        freq_shift = 0,
        flip_sin_to_cos = True,
        block_types = ("EncodeBlock3D", "EncodeAttnBlock3D", "EncodeAttnBlock3D"),
        block_channels = (224, 448, 672),
        layers_per_block = 2,
        act_fn: str = "silu",
        attention_head_dim = 8,
        norm_num_groups = 32,
        norm_eps: float = 1e-5,
        is_cond = False,
        no_time_embed = False
    ):
        super().__init__()
        self.time_embed_dim = None if no_time_embed else block_channels[0] * 4
        # Check inputs
        if len(block_channels) != len(block_types):
            raise ValueError(
                f"Must provide the same number of `block_channels` as `block_types`. `block_channels`: {block_channels}. `block_types`: {block_types}."
            )

        # input
        self.conv_in = nn.Conv3d(in_channels, block_channels[0], kernel_size=3, padding=1)

        self.encode_blocks = nn.ModuleList([])
        self.mid_block = None
        self.decode_blocks = nn.ModuleList([])

        output_channel = block_channels[0]
        for i, encode_block_type in enumerate(block_types):
            input_channel = output_channel
            output_channel = block_channels[i]

            if encode_block_type == "EncodeBlock3D":
                encode_block = EncodeBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=self.time_embed_dim,
                    num_layers=layers_per_block,
                    resnet_act_fn=act_fn,
                    num_groups=norm_num_groups,
                    eps=norm_eps
                )
            elif encode_block_type == "EncodeAttnBlock3D":
                encode_block = EncodeAttnBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=self.time_embed_dim,
                    num_layers=layers_per_block,
                    resnet_act_fn=act_fn,
                    attention_head_dim=attention_head_dim,
                    num_groups=norm_num_groups,
                    eps=norm_eps
                )
            else:
                raise ValueError(f"Unsupported encode block type: {encode_block_type}")

            self.encode_blocks.append(encode_block)

        if is_cond is False:
            # out
            num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_channels[-1] // 4, 32)
            self.conv_norm_out = nn.GroupNorm(num_channels=block_channels[-1], num_groups=num_groups_out, eps=norm_eps)
            self.conv_act = nn.SiLU()
            self.conv_out = nn.Conv3d(block_channels[-1], out_channels, kernel_size=3, padding=1)

        # time
        if no_time_embed is False:
            self.time_proj = Timesteps(block_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_channels[0]
            self.time_embedding = TimestepEmbedding(timestep_input_dim, self.time_embed_dim)
            

    def time_encode(self, timestep, bs, device):
        timesteps = timestep

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(bs, dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype, device=device)
        t_emb = self.time_embedding(t_emb)

        return t_emb

    def conv_in_model(self, sample):
        if sample.dim() == 6:
            # multiple faces in batch: bs, m, ch, n, n, n
            bs = sample.shape[0]
            m = sample.shape[1]
            sample = sample.reshape(bs * m, *sample.shape[2:])
            sample = self.conv_in(sample)
            sample = sample.reshape(bs, m, *sample.shape[1:])
        else:
            sample = self.conv_in(sample)
        return sample
    
    def encode_block_n(self, sample, t_emb, n):
        encode_block = self.encode_blocks[n]
        if sample.dim() == 6:
            # multiple faces in batch: bs, m, ch, n, n, n
            bs = sample.shape[0]
            m = sample.shape[1]
            sample = sample.reshape(bs * m, *sample.shape[2:]) # bs * m, ch, n, n, n
            # expand t_emb to match the batch size
            if t_emb is not None:
                t_emb = t_emb[:,None].repeat(1, m, 1) # bs, m, ch
                t_emb = t_emb.reshape(bs * m, *t_emb.shape[2:]) # bs * m, ch
            sample, res_samples = self.encode_block_n(sample, t_emb, n)
            sample = sample.reshape(bs, m, *sample.shape[1:]) # bs, m, ch, n, n, n
            new_res_samples = []
            for res_sample in res_samples:
                new_res_samples.append(res_sample.reshape(bs, m, *res_sample.shape[1:]))
            return sample, new_res_samples

        if hasattr(encode_block, "skip_conv"):
            sample, res_samples, skip_sample = encode_block(
                hidden_states=sample, temb=t_emb, skip_sample=skip_sample
            )
            raise NotImplementedError
        else:
            sample, res_samples = encode_block(hidden_states=sample, temb=t_emb)
        
        return sample, res_samples
    
    def conv_out_model(self, sample):
        if sample.dim() == 6:
            # multiple faces in batch: bs, m, ch, n, n, n
            bs = sample.shape[0]
            m = sample.shape[1]
            sample = sample.reshape(bs * m, *sample.shape[2:])
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
            sample = self.conv_out(sample)
            sample = sample.reshape(bs, m, *sample.shape[1:])
        else:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
            sample = self.conv_out(sample)
        return sample

    def forward(self, sample, timestep):
        # 1. time
        t_emb = self.time_encode(timestep, sample.shape[0], sample.device)

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        for encode_block_i in range(len(self.encode_blocks)):
            sample, res_samples = self.encode_block_n(sample, t_emb, encode_block_i)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        return sample

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)


