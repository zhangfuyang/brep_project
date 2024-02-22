import os
import torch
import pytorch_lightning as pl
import mcubes
import numpy as np
from torch.optim.optimizer import Optimizer
import trimesh
from diffusers import DDIMScheduler
import pickle
import torch.nn.functional as F
import time

class DiffusionExperiment(pl.LightningModule):
    def __init__(self, config, diffusion_model, face_model, sdf_model):
        super(DiffusionExperiment, self).__init__()
        self.config = config
        self.face_model = face_model
        self.sdf_model = sdf_model
        self.face_model.eval()
        self.sdf_model.eval()
        for param in self.face_model.parameters():
            param.requires_grad = False
        for param in self.sdf_model.parameters():
            param.requires_grad = False
        self.diffusion_model = diffusion_model
        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.02,
            beta_schedule='scaled_linear',
            clip_sample=False,
            set_alpha_to_one=False,
            )
        
        self.latent_dim = None
        self.test_idx = 0


    @torch.no_grad()
    def preprocess(self, batch):
        sdf_voxel = batch['sdf_voxel'] # bs, 1, N, N, N or bs, 1, 8, N, N, N
        face_voxel = batch['face_voxel'] # bs, M, N, N, N or bs, M, 8, N, N, N

        # get latent
        if sdf_voxel.dim() == 5:
            # voxel
            with torch.no_grad():
                sdf_latent = self.sdf_model.encode(sdf_voxel)[:,None] # bs, 1, C, N, N, N

                self.latent_dim = sdf_latent.shape[2]

                bs = face_voxel.shape[0]
                face_voxel = face_voxel.reshape(-1, *face_voxel.shape[2:])[:,None]
                face_latent = self.face_model.encode(face_voxel) # bs*M, C, N, N, N
                face_latent = face_latent.reshape(bs, -1, *face_latent.shape[1:])
                # bs, M, C, N, N, N
        else:
            self.latent_dim = sdf_voxel.shape[2]
            sdf_latent = sdf_voxel
            face_latent = face_voxel
        
        # normalize
        sdf_latent = (sdf_latent - self.config['solid_mean']) / self.config['solid_std']
        face_latent = (face_latent - self.config['face_mean']) / self.config['face_std']
        
        bs = sdf_latent.shape[0]
        latent = torch.cat([sdf_latent, face_latent], 1) # bs, 1+M, C, N, N, N
        
        return latent

    def training_step(self, batch, batch_idx):
        z = self.preprocess(batch)
        
        t = torch.randint(0, self.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
        noise = torch.randn_like(z[:,1:])
        noise_z = self.scheduler.add_noise(z[:,1:], noise, t)
        noise_pred = self.diffusion_model(noise_z, z[:,:1],t)
        sd_loss = torch.nn.functional.mse_loss(noise_pred, noise)
        self.log('train_loss', sd_loss, rank_zero_only=True, prog_bar=True)

        return sd_loss
    
    def on_train_epoch_end(self) -> None:
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", cur_lr, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        x = self.preprocess(batch)
        z = torch.randn_like(x[:,1:])

        self.scheduler.set_timesteps(self.config['diffusion_steps'])
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            timestep = torch.cat([t.unsqueeze(0)]*z.shape[0], 0)
            noise_pred = self.diffusion_model(z, x[:,:1], timestep)
            z = self.scheduler.step(noise_pred, t, z).prev_sample

        if self.trainer.is_global_zero:
            if batch_idx == 0: 
                for i in range(min(5, x.shape[0])):
                    sdf_voxel_gt, face_voxels_gt = self.latent_to_voxel(x[i][0], x[i][1:])
                    _, face_voxels = self.latent_to_voxel(None, z[i])

                    save_name_prefix = os.path.join(self.logger.log_dir, 'images', 
                                         f'{self.global_step}_gt')
                    self.render_mesh(sdf_voxel_gt, save_name_prefix+f'_{i}_sdf.obj', phase='sdf')

                    save_name_prefix = os.path.join(self.logger.log_dir, 'images', 
                                             f'{self.global_step}_gen')
                    self.render_mesh(face_voxels[0], save_name_prefix+f'_{i}_f0.obj', phase='face')
                    self.render_mesh(face_voxels[1], save_name_prefix+f'_{i}_f1.obj', phase='face')
                    self.render_mesh(face_voxels[2], save_name_prefix+f'_{i}_f2.obj', phase='face')
                    self.render_mesh(face_voxels[3], save_name_prefix+f'_{i}_f3.obj', phase='face')
                    self.render_mesh(face_voxels[4], save_name_prefix+f'_{i}_f4.obj', phase='face')

    def latent_to_voxel(self, sdf_latent, face_latents):
        if sdf_latent is not None:
            sdf_latent = sdf_latent[None] # 1, C, N, N, N
            sdf_latent = sdf_latent * self.config['solid_std'] + self.config['solid_mean']
            with torch.no_grad():
                sdf_voxel = self.sdf_model.quantize_decode(sdf_latent) # 1, 1, N, N, N
            sdf_voxel = sdf_voxel[0,0]
        else:
            sdf_voxel = None

        if face_latents is not None:
            face_latents = face_latents * self.config['face_std'] + self.config['face_mean']
            with torch.no_grad():
                face_voxel = self.face_model.quantize_decode(face_latents) # M, 1, N, N, N
            face_voxel = face_voxel[:,0]
        else:
            face_voxel = None
        
        return sdf_voxel, face_voxel

    def render_mesh(self, voxel, filename, phase='sdf'):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        voxel = voxel.cpu().numpy()
        if phase == 'sdf':
            vertices, triangles = mcubes.marching_cubes(voxel, 0)
            mcubes.export_obj(vertices, triangles, filename)
        elif phase == 'face':
            points = np.where(voxel < 0.02)
            points = np.array(points).T
            pointcloud = trimesh.points.PointCloud(points)
            # save
            pointcloud.export(filename)
        else:
            raise ValueError(f'phase {phase} not supported')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {'params': self.diffusion_model.voxel_unet.parameters(), 'lr': self.config['lr']},
                {'params': self.diffusion_model.face_unet.parameters(), 'lr': self.config['lr']*0.1},
                {'params': self.diffusion_model.v2f_attn.parameters(), 'lr': self.config['lr']},
                {'params': self.diffusion_model.f2f_attn.parameters(), 'lr': self.config['lr']},
            ], 
            lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['lr_decay'])
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def test_step(self, batch, batch_idx):
        x = self.preprocess(batch)
        z = torch.randn_like(x[:,1:])

        self.scheduler.set_timesteps(self.config['diffusion_steps'])
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            timestep = torch.cat([t.unsqueeze(0)]*z.shape[0], 0)
            noise_pred = self.diffusion_model(z, x[:,:1], timestep)
            z = self.scheduler.step(noise_pred, t, z).prev_sample
        
        if self.trainer.is_global_zero:
            for i in range(x.shape[0]):
                save_name = os.path.join(self.logger.log_dir, 'test', f'{self.test_idx:04d}.pkl')
                sdf_voxel_gt, _ = self.latent_to_voxel(x[i][0], None)
                _, face_voxels = self.latent_to_voxel(None, z[i])
                sdf_voxel_gt = sdf_voxel_gt.cpu().numpy()
                face_voxels = face_voxels.cpu().numpy()
                face_voxels = face_voxels.transpose(1, 2, 3, 0)
                data = {}
                data['voxel_sdf'] = sdf_voxel_gt
                data['face_bounded_distance_field'] = face_voxels
                os.makedirs(os.path.dirname(save_name), exist_ok=True)
                with open(save_name, 'wb') as f:
                    pickle.dump(data, f)

                self.test_idx += 1

