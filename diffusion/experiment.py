import os
import torch
import pytorch_lightning as pl
import mcubes
import numpy as np
from torch.optim.optimizer import Optimizer
import trimesh
from diffusers import DDIMScheduler
import pickle

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


    @torch.no_grad()
    def preprocess(self, batch):
        sdf_voxel = batch['sdf_voxel'] # bs, 1, N, N, N
        face_voxel = batch['face_voxel'] # bs, M, N, N, N

        # get latent
        with torch.no_grad():
            sdf_latent = self.sdf_model.encode(sdf_voxel)[:,None] # bs, 1, C, N, N, N

            self.latent_dim = sdf_latent.shape[2]

            bs = face_voxel.shape[0]
            face_voxel = face_voxel.reshape(-1, *face_voxel.shape[2:])[:,None]
            face_latent = self.face_model.encode(face_voxel) # bs*M, C, N, N, N
            face_latent = face_latent.reshape(bs, -1, *face_latent.shape[1:])
            # bs, M, C, N, N, N
        
        latent = torch.cat([sdf_latent, face_latent], 1) # bs, 1+M, C, N, N, N
        latent = latent.reshape(bs, -1, *latent.shape[3:])
        
        return latent

    def training_step(self, batch, batch_idx):
        z = self.preprocess(batch)
        
        t = torch.randint(0, self.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
        noise = torch.randn_like(z)
        noise_z = self.scheduler.add_noise(z, noise, t)
        noise_pred = self.diffusion_model(noise_z, t)
        sd_loss = torch.nn.functional.mse_loss(noise_pred, noise)
        self.log('train_loss', sd_loss, rank_zero_only=True, prog_bar=True)

        return sd_loss
    
    def on_train_epoch_end(self) -> None:
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", cur_lr, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        x = self.preprocess(batch)
        z = torch.randn_like(x)

        self.scheduler.set_timesteps(self.config['diffusion_steps'])
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            timestep = torch.cat([t.unsqueeze(0)]*z.shape[0], 0)
            noise_pred = self.diffusion_model(z, timestep)
            z = self.scheduler.step(noise_pred, t, z).prev_sample

        if self.trainer.is_global_zero:
            if batch_idx == 0:
                for i in range(min(2, x.shape[0])):
                    save_name_prefix = os.path.join(self.logger.log_dir, 'images', 
                                             f'{self.global_step}_gt')
                    sdf_voxel, face_voxels = self.latent_to_voxel(x[i])
                    self.render_mesh(sdf_voxel, save_name_prefix+f'_{i}_sdf.obj', phase='sdf')
                    if self.global_step == 0:
                        for face_i in range(face_voxels.shape[0]):
                            self.render_mesh(face_voxels[face_i], save_name_prefix+f'_{i}_f_{face_i}.obj', phase='face')

                    save_name_prefix = os.path.join(self.logger.log_dir, 'images', 
                                             f'{self.global_step}_recon')
                    sdf_voxel, face_voxels = self.latent_to_voxel(z[i])
                    self.render_mesh(sdf_voxel, save_name_prefix+f'_{i}_sdf.obj', phase='sdf')
                    self.render_mesh(face_voxels[0], save_name_prefix+f'_{i}_f0.obj', phase='face')
                    self.render_mesh(face_voxels[1], save_name_prefix+f'_{i}_f1.obj', phase='face')

    def on_validation_end(self):
        if self.trainer.is_global_zero:
            self.sample_images()
    
    def sample_images(self):
        pass
        
    def latent_to_voxel(self, latent):
        sdf_latent = latent[:self.latent_dim] # dim, N, N, N
        sdf_latent = sdf_latent[None]
        face_latents = latent[self.latent_dim:]
        face_latents = face_latents.reshape(-1, self.latent_dim, *face_latents.shape[1:])
        with torch.no_grad():
            sdf_voxel = self.sdf_model.quantize_decode(sdf_latent) # 1, 1, N, N, N
            face_voxel = self.face_model.quantize_decode(face_latents) # M, 1, N, N, N
        
        return sdf_voxel[0,0], face_voxel[:,0]


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
        optimizer = torch.optim.Adam(self.diffusion_model.parameters(), 
                                     lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['lr_decay'])
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def test_step(self, batch, batch_idx):
        x = self.preprocess(batch)
        z = torch.randn_like(x)

        self.scheduler.set_timesteps(self.config['diffusion_steps'])
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            timestep = torch.cat([t.unsqueeze(0)]*z.shape[0], 0)
            noise_pred = self.diffusion_model(z, timestep)
            z = self.scheduler.step(noise_pred, t, z).prev_sample
        
        if self.trainer.is_global_zero:
            for i in range(x.shape[0]):
                save_name = os.path.join(self.logger.log_dir, 'test', f'{self.test_idx:04d}.pkl')
                sdf_voxel, face_voxels = self.latent_to_voxel(z[i])
                sdf_voxel = sdf_voxel.cpu().numpy()
                face_voxels = face_voxels.cpu().numpy()
                face_voxels = face_voxels.transpose(1, 2, 3, 0)
                data = {}
                data['voxel_sdf'] = sdf_voxel
                data['face_bounded_distance_field'] = face_voxels
                os.makedirs(os.path.dirname(save_name), exist_ok=True)
                with open(save_name, 'wb') as f:
                    pickle.dump(data, f)

                self.test_idx += 1


    
