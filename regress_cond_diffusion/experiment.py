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

        # load mean and std
        with open(config['latent_std_mean_path'], 'rb') as f:
            data = pickle.load(f)
        self.face_latent_mean = data['face_mean']
        self.face_latent_std = data['face_std']
        self.solid_latent_mean = data['solid_mean']
        self.solid_latent_std = data['solid_std']

    @torch.no_grad()
    def preprocess(self, batch):
        sdf_voxel = batch['sdf_voxel'] # bs, 1, c, N, N, N
        face_cond_voxel = batch['face_cond_voxel'] # bs, M, c, N, N, N
        if 'face_target_voxel' in batch:
            face_target_voxel = batch['face_target_voxel'] # bs, c, N, N, N
        else:
            face_target_voxel = torch.zeros_like(face_cond_voxel[:,0])

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
            face_cond_latent = face_cond_voxel
            face_target_latent = face_target_voxel
        
        # normalize
        sdf_latent = (sdf_latent - self.solid_latent_mean) / self.solid_latent_std
        face_cond_latent = (face_cond_latent - self.face_latent_mean) / self.face_latent_std
        face_target_latent = (face_target_latent - self.face_latent_mean) / self.face_latent_std
        
        return face_target_latent, face_cond_latent, sdf_latent
    
    def change_pad_face_to_zero(self, latents, face_num):
        ## latents: bs, M, C, N, N, N
        ## face_num: bs
        latents = latents.clone()
        for i in range(latents.shape[0]):
            latents[i, face_num[i]:] = 0
        return latents

    def training_step(self, batch, batch_idx):
        face_target_latent, face_cond_latent, solid_latent = self.preprocess(batch)
        
        t = torch.randint(0, self.scheduler.num_train_timesteps, 
                          (face_target_latent.shape[0],), device=face_target_latent.device).long()
        noise = torch.randn_like(face_target_latent)
        noise_z = self.scheduler.add_noise(face_target_latent, noise, t)
        noise_pred = self.diffusion_model(noise_z, face_cond_latent, solid_latent, t)
        sd_loss = torch.nn.functional.mse_loss(noise_pred, noise)
        self.log('train_loss', sd_loss, rank_zero_only=True, prog_bar=True)

        return sd_loss
    
    def on_train_epoch_end(self) -> None:
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", cur_lr, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        face_target_latent, face_cond_latent, solid_latent = self.preprocess(batch)
        z = torch.randn_like(face_target_latent)

        self.scheduler.set_timesteps(self.config['diffusion_steps'])
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            timestep = torch.cat([t.unsqueeze(0)]*z.shape[0], 0)
            noise_pred = self.diffusion_model(z, face_cond_latent, solid_latent, timestep)
            z = self.scheduler.step(noise_pred, t, z).prev_sample

        if self.trainer.is_global_zero:
            if batch_idx == 0: 
                for i in range(min(4, z.shape[0])):
                    target_voxel, face_cond_voxel, solid_voxel = \
                        self.latent_to_voxel(z[i], face_cond_latent[i], solid_latent[i])
                    gt_voxel, _, _ = self.latent_to_voxel(face_target_latent[i], None, None)

                    save_name_prefix = os.path.join(self.logger.log_dir, 'images', 
                                         f'{self.global_step}')
                    self.render_mesh(solid_voxel, save_name_prefix+f'_{i}_sdf.obj', phase='sdf')

                    pc_list = []
                    base_color = np.array([0,255,0,255], dtype=np.uint8)
                    for j in range(face_cond_voxel.shape[0]):
                        pc = self.render_mesh(face_cond_voxel[j], None, phase='face', color=base_color)
                        pc_list.append(pc)
                    cond_pc = np.concatenate([pc.vertices for pc in pc_list], axis=0)
                    cond_pc = trimesh.points.PointCloud(cond_pc)
                    if cond_pc.shape[0] > 0:
                        cond_pc.colors = np.ones((cond_pc.shape[0], 4)) * base_color
                    cond_pc.export(save_name_prefix+f'_{i}_cond_faces.obj', include_color=True)
                    
                    color = np.array([255,0,0,255], dtype=np.uint8)
                    target_pc = self.render_mesh(target_voxel, None, phase='face', color=color)
                    gt_pc = self.render_mesh(gt_voxel, None, phase='face', color=color)
                    pc = np.concatenate([target_pc.vertices, cond_pc.vertices], axis=0)
                    pc = trimesh.points.PointCloud(pc)
                    if pc.shape[0] > 0:
                        pc.colors = np.concatenate((np.ones((target_pc.shape[0], 4)) * color,
                                                    np.ones((cond_pc.shape[0], 4)) * base_color), 0)
                    pc.export(save_name_prefix+f'_{i}_pred_faces.obj', include_color=True)
                    pc = np.concatenate([gt_pc.vertices, cond_pc.vertices], axis=0)
                    pc = trimesh.points.PointCloud(pc)
                    if pc.shape[0] > 0:
                        pc.colors = np.concatenate((np.ones((gt_pc.shape[0], 4)) * color,
                                                    np.ones((cond_pc.shape[0], 4)) * base_color), 0)
                    pc.export(save_name_prefix+f'_{i}_gt_faces.obj', include_color=True)

    def latent_to_voxel(self, target_latent, face_cond_latent, solid_cond_latents):
        # target_latent: C, N, N, N
        # face_cond_latent: M, C, N, N, N
        # solid_cond_latents: 1, C, N, N, N
        if target_latent is not None:
            target_latent = target_latent[None] # 1, C, N, N, N
            target_latent = target_latent * self.face_latent_std + self.face_latent_mean
            with torch.no_grad():
                target_voxel = self.face_model.quantize_decode(target_latent) # 1, 1, N, N, N
            target_voxel = target_voxel[0,0]
        else:
            target_voxel = None

        if face_cond_latent is not None:
            face_cond_latent = face_cond_latent * self.face_latent_std + self.face_latent_mean
            with torch.no_grad():
                face_cond_voxel = self.face_model.quantize_decode(face_cond_latent) # M, 1, N, N, N
            face_cond_voxel = face_cond_voxel[:,0]
        else:
            face_cond_voxel = None
        
        if solid_cond_latents is not None:
            solid_cond_latents = solid_cond_latents * self.solid_latent_std + self.solid_latent_mean
            with torch.no_grad():
                solid_voxel = self.sdf_model.quantize_decode(solid_cond_latents) # 1, 1, N, N, N
            solid_voxel = solid_voxel[0,0]
        else:
            solid_voxel = None
        
        return target_voxel, face_cond_voxel, solid_voxel

    def render_mesh(self, voxel, filename, phase='sdf',color=None):
        voxel = voxel.cpu().numpy()
        if phase == 'sdf':
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            vertices, triangles = mcubes.marching_cubes(voxel, 0)
            mcubes.export_obj(vertices, triangles, filename)
        elif phase == 'face':
            points = np.where(voxel < 0.02)
            points = np.array(points).T
            pointcloud = trimesh.points.PointCloud(points)
            if color is not None and pointcloud.shape[0] > 0:
                pointcloud.colors = np.ones((pointcloud.shape[0], 4)) * color
            # save
            if filename is None:
                return pointcloud
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            pointcloud.export(filename, include_color=True)
            return pointcloud
        else:
            raise ValueError(f'phase {phase} not supported')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {'params': self.diffusion_model.cond_solid_unet.parameters(), 'lr': self.config['lr']},
                {'params': self.diffusion_model.cond_face_unet.parameters(), 'lr': self.config['lr']},
                {'params': self.diffusion_model.face_unet.parameters(), 'lr': self.config['lr']*0.1},
                {'params': self.diffusion_model.s2f_attn.parameters(), 'lr': self.config['lr']},
                {'params': self.diffusion_model.f2f_attn.parameters(), 'lr': self.config['lr']},
            ], 
            lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['lr_decay'])
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def test_step(self, batch, batch_idx):
        face_num = 0

        face_target_latent, face_cond_latent, solid_latent = self.preprocess(batch)

        self.scheduler.set_timesteps(self.config['diffusion_steps'])
        timesteps = self.scheduler.timesteps

        all_face_cond_latent = [face_cond_latent[0,0]]
                    
        _,_,solid_voxel = \
            self.latent_to_voxel(None, None, solid_latent[0])
        save_name_prefix = os.path.join(self.logger.log_dir, 'test', 
                             f'{self.test_idx:04d}')
        self.render_mesh(solid_voxel, save_name_prefix+f'/sdf.obj', phase='sdf')

        for round_i in range(15):
            z = torch.randn_like(face_target_latent)
            for i, t in enumerate(timesteps):
                timestep = torch.cat([t.unsqueeze(0)]*z.shape[0], 0)
                noise_pred = self.diffusion_model(z, face_cond_latent, solid_latent, timestep)
                z = self.scheduler.step(noise_pred, t, z).prev_sample
        
            target_voxel, face_cond_voxel, solid_voxel = \
                self.latent_to_voxel(z[0], face_cond_latent[0], solid_latent[0])
            
            all_face_cond_latent.append(z[0])

            target_voxel_valid = target_voxel < 0.03
            face_cond_voxel_valid = (face_cond_voxel < 0.03).sum(0)
            solid_voxel_valid = torch.abs(solid_voxel) < 0.03
            # check percentage of covering new area in solid
            already = torch.logical_and(face_cond_voxel_valid>0, solid_voxel_valid)
            new = torch.logical_and(target_voxel_valid, torch.logical_and(solid_voxel_valid, ~already))
            new_cover_percentage = (new.sum() / (target_voxel_valid.sum()+1e-5)).item()
            overlaping_percentage = (torch.logical_and(target_voxel_valid, face_cond_voxel_valid>0).sum() / (target_voxel_valid.sum()+1e-5)).item()
            good_cover_percentage = (torch.logical_and(target_voxel_valid, solid_voxel_valid).sum() / (target_voxel_valid.sum()+1e-5)).item()

            if target_voxel_valid.sum() < 5:
                overall_coverage = torch.logical_and(face_cond_voxel_valid>0, solid_voxel_valid).sum() / (solid_voxel_valid.sum()+1e-5)
                if torch.rand(1).item() < overall_coverage:
                    break

            if torch.rand(1).item() < new_cover_percentage and \
                torch.rand(1).item() < good_cover_percentage:
                print('find new face')
                face_cond_latent = torch.cat((face_cond_latent, z[None]), 1)
                face_num += 1
                if torch.rand(1).item() < overlaping_percentage:
                    print('remove one face')
                    remove_idx = torch.randint(1, face_cond_latent.shape[1], (1,)).item()
                    face_cond_latent = torch.cat((face_cond_latent[:,:remove_idx], face_cond_latent[:,remove_idx+1:]), 1)
                    face_num -= 1
            
            # save fig
            pc_list = []
            base_color = np.array([0,255,0,255], dtype=np.uint8)
            for j in range(face_cond_voxel.shape[0]):
                pc = self.render_mesh(face_cond_voxel[j], None, phase='face', color=base_color)
                pc_list.append(pc)
            cond_pc = np.concatenate([pc.vertices for pc in pc_list], axis=0)
            cond_pc = trimesh.points.PointCloud(cond_pc)
            if cond_pc.shape[0] > 0:
                cond_pc.colors = np.ones((cond_pc.shape[0], 4)) * base_color
            save_name = os.path.join(self.logger.log_dir, 'test', 
                                 f'{self.test_idx:04d}', f'{round_i:02d}_cond.obj')
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            cond_pc.export(save_name, include_color=True)
            
            color = np.array([255,0,0,255], dtype=np.uint8)
            target_pc = self.render_mesh(target_voxel, None, phase='face', color=color)
            pc = np.concatenate([target_pc.vertices, cond_pc.vertices], axis=0)
            pc = trimesh.points.PointCloud(pc)
            if pc.shape[0] > 0:
                pc.colors = np.concatenate((np.ones((target_pc.shape[0], 4)) * color,
                                            np.ones((cond_pc.shape[0], 4)) * base_color), 0)
            save_name = os.path.join(self.logger.log_dir, 'test', 
                                 f'{self.test_idx:04d}', f'{round_i:02d}_pred.obj')
            pc.export(save_name, include_color=True)
            

        all_face_cond_latent = torch.stack(all_face_cond_latent, 0) 
        if self.trainer.is_global_zero:
            _, face_voxel, solid_voxel = \
                self.latent_to_voxel(None, face_cond_latent[0], solid_latent[0])
            solid_voxel = solid_voxel.cpu().numpy()
            face_voxel = face_voxel.cpu().numpy()
            face_voxel = face_voxel.transpose(1, 2, 3, 0)
            data = {}
            data['voxel_sdf'] = solid_voxel
            data['face_bounded_distance_field'] = face_voxel
            save_name = os.path.join(self.logger.log_dir, 'test', 
                                     f'{self.test_idx:04d}', f'pred.pkl')
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            with open(save_name, 'wb') as f:
                pickle.dump(data, f)
            _, all_face_voxel, _ = self.latent_to_voxel(None, all_face_cond_latent, None)
            all_face_voxel = all_face_voxel.cpu().numpy()
            all_face_voxel = all_face_voxel.transpose(1, 2, 3, 0)
            data = {}
            data['voxel_sdf'] = solid_voxel
            data['face_bounded_distance_field'] = all_face_voxel
            save_name = os.path.join(self.logger.log_dir, 'test', 
                                     f'{self.test_idx:04d}', 'all.pkl')
            with open(save_name, 'wb') as f:
                pickle.dump(data, f)

            # call data_render_mc.py
            command = f'python data_rendering_mc.py --data_root ' + \
                        f'{os.path.join(self.logger.log_dir, "test")} --folder_name {self.test_idx:04d} ' + \
                        f'--save_root {os.path.join(self.logger.log_dir, "test", f"{self.test_idx:04d}", "test_render")}'
            print(f'now running {command}')
            os.system(command)

            self.test_idx += 1
        
    
    def on_test_end(self) -> None:
        return
        # call data_render_mc.py
        command = f'python data_rendering_mc.py --data_root {self.logger.log_dir} --folder_name test'
        print(f'now running {command}')
        os.system(command)

