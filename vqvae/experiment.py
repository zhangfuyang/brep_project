import os
import torch
import pytorch_lightning as pl
import mcubes
import numpy as np
import trimesh
import pickle

class VAEExperiment(pl.LightningModule):
    def __init__(self, config, vae_model):
        super(VAEExperiment, self).__init__()
        self.config = config
        self.vae_model = vae_model
    
    def training_step(self, batch, batch_idx):
        batch, data_name = batch[0], batch[1]
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
                for i in range(min(4, len(batch))):
                    save_name = os.path.join(self.logger.log_dir, 'images', 
                                             f'{self.global_step}_train_gt_{i}.obj')
                    self.render_mesh(batch[i][0], save_name)

                    save_name = os.path.join(self.logger.log_dir, 'images', 
                                             f'{self.global_step}_train_recon_{i}.obj')
                    self.render_mesh(recon[i][0].detach(), save_name)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch, data_name = batch[0], batch[1]
        results = self.vae_model(batch, self.config['vq_weight'])
        recon = results[0]
        vq_loss = results[1]

        loss_dict = self.vae_model.loss_function(recon, batch, vq_loss, **self.config)
        loss = loss_dict['loss']
        self.log('val_loss', loss, rank_zero_only=True, prog_bar=True)
        if self.trainer.is_global_zero:
            if batch_idx == 0:
                for i in range(min(10, len(batch))):
                    save_name = os.path.join(self.logger.log_dir, 'images', 
                                             f'{self.global_step}_gt_{i}.obj')
                    self.render_mesh(batch[i][0], save_name)

                    save_name = os.path.join(self.logger.log_dir, 'images', 
                                             f'{self.global_step}_recon_{i}.obj')
                    self.render_mesh(recon[i][0], save_name)

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
            

class ReconVAEExperiment(pl.LightningModule):
    def __init__(self, config, solid_model, face_model):
        super(ReconVAEExperiment, self).__init__()
        self.config = config
        self.solid_model = solid_model
        self.face_model = face_model
        self.solid_model.eval()
        self.face_model.eval()
    
    def validation_step(self, batch, batch_idx):
        solid_voxel, faces_voxel, filenames, faces_num = \
            batch['solid_voxel'], batch['faces_voxel'], batch['filename'], batch['faces_num']
        
        solid_latent = self.solid_model.encode(solid_voxel) # bs, 8, N, N, N
        # minibatch for face model
        faces_mini_batch = 128
        face_latent = []
        for i in range(0, faces_voxel.shape[0], faces_mini_batch):
            result = self.face_model.encode(faces_voxel[i:i+faces_mini_batch])
            face_latent.append(result)
        face_latent = torch.cat(face_latent, dim=0) # ?, 8, N, N, N

        # save pkl
        face_num_start = 0
        for i in range(solid_voxel.shape[0]):
            filename = filenames[i]
            save_path = os.path.join(self.logger.log_dir, 'pkl', f'{filename}.pkl')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            data = {}
            data['voxel_sdf'] = solid_latent[i].cpu().numpy() # 8, N, N, N
            num = faces_num[i]
            face_result = face_latent[face_num_start:face_num_start+num] #?, 8, N, N, N
            face_result = face_result.permute(1, 2, 3, 4, 0) #8, N, N, N, ?
            data['face_bounded_distance_field'] = face_result.cpu().numpy()
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
            
            face_num_start += num

