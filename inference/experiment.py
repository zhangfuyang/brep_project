import os
import torch
import pytorch_lightning as pl
import mcubes
import numpy as np
import trimesh

class VAEExperiment(pl.LightningModule):
    def __init__(self, config, vae_model):
        super(VAEExperiment, self).__init__()
        self.config = config
        self.vae_model = vae_model
    
    def training_step(self, batch, batch_idx):
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
        torch.cuda.empty_cache()
        quantize_indices = self.vae_model.quantize_indices(batch['x'])
        # bs, N, N, N
        # save quantize indices
        for bs_i in range(quantize_indices.shape[0]):
            save_name = os.path.join(self.logger.log_dir, 'quantization', 
                                     f'{batch["filename"][bs_i]}.npy')
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            np.save(save_name, quantize_indices[bs_i].cpu().numpy())

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
        optimizer = torch.optim.Adam(self.vae_model.parameters(), 
                                     lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['lr_decay'])
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
            
    
