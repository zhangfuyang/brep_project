import os
import torch
import pytorch_lightning as pl
import mcubes
import numpy as np
import trimesh
from diffusers import DDIMScheduler

class DiffusionExperiment(pl.LightningModule):
    def __init__(self, config, diffusion_model, face_model, sdf_model):
        super(DiffusionExperiment, self).__init__()
        self.config = config
        self.face_model = face_model
        self.sdf_model = sdf_model
        self.face_model.eval()
        self.sdf_model.eval()
        self.diffusion_model = diffusion_model
        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.02,
            beta_schedule='scaled_linear',
            clip_sample=False,
            set_alpha_to_one=False,
            )
    
    def preprocess(self, batch):
        sdf_voxel = batch['sdf_voxel'] # bs, 1, N, N, N
        face_voxel = batch['face_voxel'] # bs, M, N, N, N

        # get latent
        with torch.no_grad():
            sdf_latent = self.sdf_model.encode(sdf_voxel)[:,None] # bs, 1, C, N, N, N

            bs = face_voxel.shape[0]
            face_voxel = face_voxel.reshape(-1, *face_voxel.shape[2:])[:,None]
            face_latent = self.face_model.encode(face_voxel) # bs*M, C, N, N, N
            face_latent = face_latent.reshape(bs, -1, *face_latent.shape[1:])
            # bs, M, C, N, N, N
        
        return sdf_latent, face_latent

    
    def training_step(self, batch, batch_idx):
        sdf_latent, face_latent = self.preprocess(batch)

        results = self.vae_model(batch, self.config['vq_weight'])
        recon = results[0]
        vq_loss = results[1]

        loss_dict = self.vae_model.loss_function(
            recon, batch, vq_loss, **self.config)
        loss = loss_dict['loss']
        recon_loss = loss_dict['Reconstruction_Loss']
        vq_loss = loss_dict['VQ_Loss']
        self.log('recon_loss', recon_loss, rank_zero_only=True, prog_bar=True)
        self.log('vq_loss', vq_loss, rank_zero_only=True, prog_bar=True)
        self.log('train_loss', loss, rank_zero_only=True, prog_bar=True)
        if self.trainer.is_global_zero and batch_idx == 0:
            if batch_idx == 0:
                for i in range(4):
                    save_name = os.path.join(self.logger.log_dir, 'images', 
                                             f'{self.global_step}_train_gt_{i}.obj')
                    self.render_mesh(batch[i][0], save_name)

                    save_name = os.path.join(self.logger.log_dir, 'images', 
                                             f'{self.global_step}_train_recon_{i}.obj')
                    self.render_mesh(recon[i][0].detach(), save_name)
        return loss
    
    def validation_step(self, batch, batch_idx):
        sdf_latent, face_latent = self.preprocess(batch)

        x = torch.cat([sdf_latent, face_latent], 1) # bs, 1+M, C, N, N, N
        z = torch.randn_like(x)
        face_num = batch['face_num']

        self.scheduler.set_timesteps(self.config['diffusion_steps'])
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            timestep = torch.cat([t.unsqueeze(0)]*z.shape[0], 0)
            noise_pred = self.diffusion_model(z, timestep, face_num)
            z = self.scheduler.step(noise_pred, t, z).prev_sample

        if self.trainer.is_global_zero:
            if batch_idx == 0:
                for i in range(4):
                    save_name = os.path.join(self.logger.log_dir, 'images', 
                                             f'{self.global_step}_gt_{i}.obj')
                    self.render_mesh(batch[i][0], save_name)

                    save_name = os.path.join(self.logger.log_dir, 'images', 
                                             f'{self.global_step}_recon_{i}.obj')
                    self.render_mesh(recon[i][0], save_name)

    def on_validation_end(self):
        if self.trainer.is_global_zero:
            self.sample_images()
    
    def sample_images(self):
        pass
        
    def render_mesh(self, voxel, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        voxel = voxel.cpu().numpy()
        #voxel = voxel / 3
        phase = self.config['phase']
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
            
    
