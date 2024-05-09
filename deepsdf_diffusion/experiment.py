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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class DiffusionExperiment(pl.LightningModule):
    def __init__(self, config, diffusion_model, face_model, solid_model):
        super(DiffusionExperiment, self).__init__()
        self.config = config
        self.face_model = face_model
        self.solid_model = solid_model
        if self.face_model is not None:
            self.face_model.eval()
            for param in self.face_model.parameters():
                param.requires_grad = False
        if self.solid_model is not None:
            self.solid_model.eval()
            for param in self.solid_model.parameters():
                param.requires_grad = False
        self.diffusion_model = diffusion_model
        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.02,
            beta_schedule='scaled_linear',
            clip_sample=False,
            set_alpha_to_one=False,
            prediction_type=self.config['prediction_type'],
            )
        
        self.latent_dim = None
        self.test_idx = 0

        self.face_mean_std = None
        self.solid_mean_std = None

    @torch.no_grad()
    def preprocess(self, batch):
        if self.face_mean_std is None:
            self.face_mean_std = (batch['face_mean'][0], batch['face_std'][0])
        if self.solid_mean_std is None:
            self.solid_mean_std = (batch['solid_mean'][0], batch['solid_std'][0])
        solid_latent = batch['solid_latent'] # bs, C
        solid_latent = solid_latent.unsqueeze(1) # bs, 1, C
        face_latents = batch['face_latents'] # bs, M, C

        # normalize
        solid_latent = (solid_latent - self.solid_mean_std[0]) / self.solid_mean_std[1]
        face_latents = (face_latents - self.face_mean_std[0]) / self.face_mean_std[1]

        return solid_latent, face_latents
    
    def training_step(self, batch, batch_idx):
        solid_latent, face_latents = self.preprocess(batch)
        
        t = torch.randint(0, self.scheduler.num_train_timesteps, (solid_latent.shape[0],), device=solid_latent.device).long()
        if self.config['generate_solid'] is False:
            noise = torch.randn_like(face_latents)
            noise_face = self.scheduler.add_noise(face_latents, noise, t)
            model_output_face, _ = self.diffusion_model(noise_face, solid_latent, t)
            model_output = model_output_face
            if self.config['prediction_type'] == 'sample':
                target = face_latents
            elif self.config['prediction_type'] == 'epsilon':
                target = noise
        else:
            # generate both faces and solid
            noise1 = torch.randn_like(face_latents)
            noise_face = self.scheduler.add_noise(face_latents, noise1, t)
            noise2 = torch.randn_like(solid_latent)
            noise_solid = self.scheduler.add_noise(solid_latent, noise2, t)
            model_output_face, model_output_solid = self.diffusion_model(noise_face, noise_solid, t)
            model_output = torch.cat([model_output_solid, model_output_face], 1)
            if self.config['prediction_type'] == 'sample':
                target = torch.cat([solid_latent, face_latents], 1)
            elif self.config['prediction_type'] == 'epsilon':
                target = torch.cat([noise2, noise1], 1)
            
        if self.config['loss_type'] == 'l2':
            sd_loss = torch.nn.functional.mse_loss(model_output, target)
        elif self.config['loss_type'] == 'l1':
            sd_loss = torch.nn.functional.l1_loss(model_output, target)

        self.log('train_loss', sd_loss, rank_zero_only=True, prog_bar=True)

        return sd_loss
    
    def on_train_epoch_end(self) -> None:
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", cur_lr, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        solid_latent, face_latents = self.preprocess(batch)
        z_faces = torch.randn_like(face_latents) # bs, M, C
        if self.config['generate_solid']:
            z_solid = torch.randn_like(solid_latent) # bs, 1, C
        else:
            z_solid = solid_latent

        self.scheduler.set_timesteps(self.config['diffusion_steps'])
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            timestep = torch.cat([t.unsqueeze(0)]*z_faces.shape[0], 0)
            if self.config['generate_solid']:
                model_output_face, model_output_solid = self.diffusion_model(z_faces, z_solid, timestep)
                z_solid = self.scheduler.step(model_output_solid, t, z_solid).prev_sample
                z_faces = self.scheduler.step(model_output_face, t, z_faces).prev_sample
            else:
                model_output_face, _ = self.diffusion_model(z_faces, z_solid, timestep)
                z_faces = self.scheduler.step(model_output_face, t, z_faces).prev_sample
        
        if self.trainer.is_global_zero:
            for i in range(solid_latent.shape[0]):
                # copy brep
                solid_path = batch['solid_path'][i]
                solid_path = solid_path.replace('.npy', '.stl')
                save_name = os.path.join(self.logger.log_dir, 'images', f'{self.global_step}_{i}.stl')
                os.makedirs(os.path.dirname(save_name), exist_ok=True)
                os.system(f'cp {solid_path} {save_name}')
                ### gt
                self.render_(solid_latent[i], face_latents[i],
                             os.path.join(self.logger.log_dir, 'images', f'{self.global_step}_{i}'),
                             '_gt')
                ### pred
                #self.render_(z_solid[i], z_faces[i],
                #             os.path.join(self.logger.log_dir, 'images', f'{self.global_step}_{i}'),
                #             '_pred')

    def render_(self, solid_code, face_codes, save_name_prefix, save_name_postfix=''):
        # solid_code: (1, C)
        # face_codes: (M, C)
        solid_code = solid_code * self.solid_mean_std[1] + self.solid_mean_std[0]
        face_codes = face_codes * self.face_mean_std[1] + self.face_mean_std[0]

        # nms
        if True:
            valid_face_codes = []
            for i in range(face_codes.shape[0]):
                good_flag = True
                for valid_code in valid_face_codes:
                    if (face_codes[i] - valid_code).abs().sum() < 10:
                        good_flag = False
                        break
                if good_flag:
                    valid_face_codes.append(face_codes[i])
            face_codes = torch.stack(valid_face_codes, 0)


        reso = 128
        line = torch.linspace(-1, 1, reso)
        xyz = torch.stack(torch.meshgrid(line, line, line), -1)
        points = xyz.reshape(-1, 3) # reso^3, 3
        points = points.to(solid_code.device)

        random_points = torch.rand(10000, 3).to(solid_code.device) * 2 - 1 # [-1, 1]

        # 1. solid
        # use group
        group_size = 10000
        solid_code = solid_code.expand(group_size, -1)
        for i in range(0, points.shape[0], group_size):
            tmp_points = points[i:i+group_size]
            pred = self.solid_model(tmp_points, solid_code[:tmp_points.shape[0]])
            if i == 0:
                pred_all = pred
            else:
                pred_all = torch.cat([pred_all, pred], 0)
        pred = pred_all
        pred = pred[:,0].reshape(reso, reso, reso)
        pred = pred.cpu().numpy() # reso, reso, reso
        vertices, triangles = mcubes.marching_cubes(pred, 0)
        save_name = save_name_prefix+'_solid'+save_name_postfix+'.obj'
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        mcubes.export_obj(vertices, triangles, save_name)

        # 2. face
        random_points_np = random_points.cpu().numpy()
        for j in range(face_codes.shape[0]):
            face_code = face_codes[j].unsqueeze(0)
            face_code = face_code.expand(random_points.shape[0], -1)
            pred = self.face_model(random_points, face_code)
            pred = pred.cpu().numpy()
            
            colors = []
            for i in range(len(random_points_np)):
                dist = pred[i]
                if dist < -0.05:
                    colors.append([0, 255, 0, 255])
                elif dist > 0.05:
                    colors.append([0, 0, 255, 255])
                else:
                    colors.append([255, 0, 0, 255])
            colors = np.array(colors).astype(np.uint8)
            pc = trimesh.points.PointCloud(random_points_np, colors=colors)
            # save
            save_name = save_name_prefix+f'_face_{j}'+save_name_postfix+'.obj'
            pc.export(save_name, include_color=True)
        
        # 3. solid + face
        # vertices [0, reso] -> [-1, 1]
        base_color = np.array(
            [[255,   0,  0, 255],  # Red
            [  0, 255,   0, 255],  # Green
            [  0,   0, 255, 255],  # Blue
            [255, 255,   0, 255],  # Yellow
            [  0, 255, 255, 255],  # Cyan
            [255,   0, 255, 255],  # Magenta
            [255, 165,   0, 255],  # Orange
            [128,   0, 128, 255],  # Purple
            [255, 192, 203, 255],  # Pink
            [128, 128, 128, 255],  # Gray
            [210, 245, 60, 255], # Lime
            [170, 110, 40, 255], # Brown
            [128, 0, 0, 255], # Maroon
            [0, 128, 128, 255], # Teal
            [0, 0, 128, 255], # Navy
            ],
            dtype=np.uint8
        )

        vertices = vertices / (reso-1) * 2 - 1
        vertices = torch.tensor(vertices).float().to(solid_code.device) # V, 3

        all_pred = []
        for j in range(face_codes.shape[0]):
            face_code = face_codes[j].unsqueeze(0)
            face_code = face_code.expand(vertices.shape[0], -1)
            pred = self.face_model(vertices, face_code)[:,0] # V
            all_pred.append(pred)
        
        all_pred = torch.stack(all_pred, 0) # M, V
        v_face_id = torch.argmin(all_pred, 0) # V
        v_face_id = v_face_id.cpu().numpy()

        v_color = base_color[v_face_id % base_color.shape[0]]
        if vertices.shape[0] > 0:
            pc = trimesh.points.PointCloud(vertices.cpu().numpy(), colors=v_color)
        save_name = save_name_prefix+'_final'+save_name_postfix+'.obj'
        pc.export(save_name, include_color=True)
                
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {'params': self.diffusion_model.parameters(), 'lr': self.config['lr']},
            ], 
            lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['lr_decay'])
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        x = self.preprocess(batch)
        z_faces = torch.randn_like(x[:,1:])
        if self.config['generate_solid']:
            z_solid = torch.randn_like(x[:,:1])
        else:
            z_solid = x[:,:1]

        self.scheduler.set_timesteps(self.config['diffusion_steps'])
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            timestep = torch.cat([t.unsqueeze(0)]*z_faces.shape[0], 0)
            if self.config['generate_solid']:
                noise_pred_face, noise_pred_solid = self.diffusion_model(z_faces, z_solid, timestep)
                z_solid = self.scheduler.step(noise_pred_solid, t, z_solid).prev_sample
                z_faces = self.scheduler.step(noise_pred_face, t, z_faces).prev_sample
            else:
                noise_pred_face, _ = self.diffusion_model(z_faces, z_solid, timestep)
                z_faces = self.scheduler.step(noise_pred_face, t, z_faces).prev_sample

        if self.trainer.is_global_zero:
            if batch_idx == 0: 
                for i in range(min(20, x.shape[0])):
                    sdf_voxel_gt, face_voxels_gt = self.latent_to_voxel(x[i][0], x[i][1:])
                    sdf_voxel, face_voxels = self.latent_to_voxel(z_solid[i][0], z_faces[i])

                    sdf_voxel_gt = sdf_voxel_gt.cpu().numpy()
                    sdf_voxel = sdf_voxel.cpu().numpy()
                    face_voxels = face_voxels.cpu().numpy()
                    face_voxels = face_voxels.transpose(1, 2, 3, 0)
                    face_voxels_gt = face_voxels_gt.cpu().numpy()
                    face_voxels_gt = face_voxels_gt.transpose(1, 2, 3, 0)
                    data = {}
                    data['voxel_sdf'] = sdf_voxel
                    data['face_bounded_distance_field'] = face_voxels
                    save_name = os.path.join(self.logger.log_dir, 'test', f'{self.test_idx:04d}.pkl')
                    os.makedirs(os.path.dirname(save_name), exist_ok=True)
                    with open(save_name, 'wb') as f:
                        pickle.dump(data, f)

                    data = {}
                    data['voxel_sdf'] = sdf_voxel_gt
                    data['face_bounded_distance_field'] = face_voxels_gt
                    save_name = os.path.join(self.logger.log_dir, 'test', f'{self.test_idx:04d}_gt.pkl')
                    os.makedirs(os.path.dirname(save_name), exist_ok=True)
                    with open(save_name, 'wb') as f:
                        pickle.dump(data, f)

                    self.test_idx += 1
    
    def on_test_end(self) -> None:
        # call data_render_mc.py
        command = f'python data_rendering_mc.py --data_root {self.logger.log_dir} --folder_name test'
        os.system(command)
        command = f'python data_rendering_mc.py --data_root '\
                  f'{self.logger.log_dir} --folder_name test --save_root {os.path.join(self.logger.log_dir, "test_render_nms")} --apply_nms'
        print(f'now running {command}')
        os.system(command)

