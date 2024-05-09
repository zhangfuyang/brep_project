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
    def __init__(self, config, diffusion_model, face_model, sdf_model):
        super(DiffusionExperiment, self).__init__()
        self.config = config
        self.face_model = face_model
        self.sdf_model = sdf_model
        if self.face_model is not None:
            self.face_model.eval()
            for param in self.face_model.parameters():
                param.requires_grad = False
        if self.sdf_model is not None:
            self.sdf_model.eval()
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
            prediction_type=self.config['prediction_type'],
            )
        
        self.latent_dim = None
        self.test_idx = 0

        # load mean and std
        if config['latent_std_mean_path'] is not None:
            with open(config['latent_std_mean_path'], 'rb') as f:
                data = pickle.load(f)
            self.face_latent_mean = data['face_mean']
            self.face_latent_std = data['face_std']
            self.solid_latent_mean = data['solid_mean']
            self.solid_latent_std = data['solid_std']
        else:
            self.face_latent_mean = 0
            self.face_latent_std = 1
            self.solid_latent_mean = 0
            self.solid_latent_std = 1

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
        sdf_latent = (sdf_latent - self.solid_latent_mean) / self.solid_latent_std
        face_latent = (face_latent - self.face_latent_mean) / self.face_latent_std

        bs = sdf_latent.shape[0]
        latent = torch.cat([sdf_latent, face_latent], 1) # bs, 1+M, C, N, N, N

        return latent
    
    def training_step(self, batch, batch_idx):
        z = self.preprocess(batch)
        
        t = torch.randint(0, self.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
        if self.config['generate_solid'] is False:
            noise = torch.randn_like(z[:,1:])
            noise_z = self.scheduler.add_noise(z[:,1:], noise, t)
            model_output_face, _ = self.diffusion_model(noise_z, z[:,:1],t)
            model_output = model_output_face
            if self.config['prediction_type'] == 'sample':
                target = z[:,1:]
            elif self.config['prediction_type'] == 'epsilon':
                target = noise
        else:
            # generate both faces and solid
            noise = torch.randn_like(z)
            noise_z = self.scheduler.add_noise(z, noise, t)
            model_output_face, model_output_solid = self.diffusion_model(noise_z[:,1:], noise_z[:,:1],t)
            model_output = torch.cat([model_output_solid, model_output_face], 1)
            if self.config['prediction_type'] == 'sample':
                target = z
            elif self.config['prediction_type'] == 'epsilon':
                target = noise
            
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
                model_output_face, model_output_solid = self.diffusion_model(z_faces, z_solid, timestep)
                z_solid = self.scheduler.step(model_output_solid, t, z_solid).prev_sample
                z_faces = self.scheduler.step(model_output_face, t, z_faces).prev_sample
            else:
                model_output_face, _ = self.diffusion_model(z_faces, z_solid, timestep)
                z_faces = self.scheduler.step(model_output_face, t, z_faces).prev_sample
        
        def nms(f_bdf):
            # f_bdf: (M, 64, 64, 64)
            # NMS
            def similarity(a, b, threshold=0.03):
                A = torch.abs(a) < threshold
                B = torch.abs(b) < threshold
                return torch.logical_and(A,B).sum() / (torch.logical_or(A,B).sum() + 1e-8)
            
            def coverage(a, b):
                A = torch.abs(a) < 0.03
                B = torch.abs(b) < 0.03
                return torch.logical_and(A,B).sum() / (A.sum() + 1e-8)

            valid_idx = []
            for i in range(1, f_bdf.shape[0]):
                is_valid = True
                for j in valid_idx:
                    if similarity(f_bdf[i], f_bdf[j]) > 0.2:
                        is_valid = False
                        break
                    coverage_ij = coverage(f_bdf[i], f_bdf[j])
                    coverage_ji = coverage(f_bdf[j], f_bdf[i])
                    if coverage_ij > 0.5 or coverage_ji > 0.5:
                        if coverage_ij > coverage_ji:
                            is_valid = False
                            break
                        else:
                            valid_idx.remove(j)
                if is_valid:
                    valid_idx.append(i)
            return f_bdf[valid_idx]

        def plot_3d_bbox(ax, min_corner, max_corner, color='r'):
            vertices = [
                (min_corner[0], min_corner[1], min_corner[2]),
                (max_corner[0], min_corner[1], min_corner[2]),
                (max_corner[0], max_corner[1], min_corner[2]),
                (min_corner[0], max_corner[1], min_corner[2]),
                (min_corner[0], min_corner[1], max_corner[2]),
                (max_corner[0], min_corner[1], max_corner[2]),
                (max_corner[0], max_corner[1], max_corner[2]),
                (min_corner[0], max_corner[1], max_corner[2])
            ]

            # Define the 12 triangles composing the box
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]], 
                [vertices[0], vertices[1], vertices[5], vertices[4]], 
                [vertices[2], vertices[3], vertices[7], vertices[6]], 
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[4], vertices[7], vertices[3], vertices[0]]
            ]

            ax.add_collection3d(Poly3DCollection(faces, facecolors='blue', linewidths=1, edgecolors=color, alpha=0))
        
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
        if self.trainer.is_global_zero:
            if batch_idx == 0: 
                if self.config['render_type'] == 'bbox':
                    for i in range(min(400, x.shape[0])):
                        colors = cm.rainbow(np.linspace(0, 1, x[:,1:].shape[1]))
                        np.random.shuffle(colors)

                        # gt
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        ax.set_xlim([-1.1, 1.1])  
                        ax.set_ylim([-1.1, 1.1])  
                        ax.set_zlim([-1.1, 1.1]) 
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_zticks([])
                        gt_bbox = x[i, 1:, :, 0,0,0].cpu().numpy() # M, 6
                        for j in range(gt_bbox.shape[0]):
                            min_corner = gt_bbox[j, :3] - gt_bbox[j, 3:] / 2
                            max_corner = gt_bbox[j, :3] + gt_bbox[j, 3:] / 2
                            plot_3d_bbox(ax, min_corner, max_corner, color=colors[j])
                        
                        save_name = os.path.join(self.logger.log_dir, 'images',
                                            f'{self.global_step}_{i}_gt.png')
                        os.makedirs(os.path.dirname(save_name), exist_ok=True)
                        plt.savefig(save_name)
                        plt.close()
                        gt_solid_path = batch['filename'][i]
                        os.system(f'cp {gt_solid_path} {os.path.join(self.logger.log_dir, "images", f"{self.global_step}_{i}_gt_solid.stl")}')

                        # pred
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        ax.set_xlim([-1.1, 1.1])  
                        ax.set_ylim([-1.1, 1.1])  
                        ax.set_zlim([-1.1, 1.1]) 
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_zticks([])
                        pred_bbox = z_faces[i, :, :, 0,0,0].cpu().numpy() # M, 6
                        for j in range(pred_bbox.shape[0]):
                            min_corner = pred_bbox[j, :3] - pred_bbox[j, 3:] / 2
                            max_corner = pred_bbox[j, :3] + pred_bbox[j, 3:] / 2
                            plot_3d_bbox(ax, min_corner, max_corner, color=colors[j])
                        
                        save_name = os.path.join(self.logger.log_dir, 'images',
                                            f'{self.global_step}_{i}.png')
                        os.makedirs(os.path.dirname(save_name), exist_ok=True)
                        plt.savefig(save_name)
                        plt.close()

                        # save in text
                        with open(os.path.join(self.logger.log_dir, 'images', f'{self.global_step}_{i}.txt'), 'w') as f:
                            for j in range(gt_bbox.shape[0]):
                                f.write(' '.join([str(x) for x in gt_bbox[j]]))
                                f.write('\n')
                            f.write('\n')
                            f.write('\n\n')
                            for j in range(pred_bbox.shape[0]):
                                f.write(' '.join([str(x) for x in pred_bbox[j]]))
                                f.write('\n')
                    return

                for i in range(min(400, x.shape[0])):
                    sdf_voxel_gt, face_voxels_gt = self.latent_to_voxel(x[i][0], x[i][1:])
                    sdf_voxel, face_voxels = self.latent_to_voxel(z_solid[i][0], z_faces[i])

                    face_voxels = nms(face_voxels)
                    save_name_prefix = os.path.join(self.logger.log_dir, 'images', 
                                         f'{self.global_step}')
                    self.render_mesh(sdf_voxel_gt, save_name_prefix+f'_{i}_sdf_gt.obj', phase='sdf')
                    if self.config['generate_solid']:
                        self.render_mesh(sdf_voxel, save_name_prefix+f'_{i}_sdf.obj', phase='sdf')

                    all_pc = []
                    all_color = []
                    for face_i in range(face_voxels.shape[0]):
                        face_pc = self.render_mesh(face_voxels[face_i], None, phase='face')
                        pc = face_pc.vertices
                        color = np.ones((pc.shape[0], 4), dtype=np.uint8) * base_color[face_i % base_color.shape[0]]
                        all_pc.append(pc)
                        all_color.append(color)
                    all_pc = np.concatenate(all_pc, 0)
                    all_color = np.concatenate(all_color, 0)
                    if all_pc.shape[0] > 0:
                        pointcloud = trimesh.points.PointCloud(all_pc, colors=all_color)
                        pointcloud.export(save_name_prefix+f'_{i}_face.obj', include_color=True)

                    all_pc = []
                    all_color = []
                    for face_i in range(face_voxels_gt.shape[0]):
                        face_pc = self.render_mesh(face_voxels_gt[face_i], None, phase='face')
                        pc = face_pc.vertices
                        color = np.ones((pc.shape[0], 4), dtype=np.uint8) * base_color[face_i % base_color.shape[0]]
                        all_pc.append(pc)
                        all_color.append(color)
                    all_pc = np.concatenate(all_pc, 0)
                    all_color = np.concatenate(all_color, 0)
                    if all_pc.shape[0] > 0:
                        pointcloud = trimesh.points.PointCloud(all_pc, colors=all_color)
                        pointcloud.export(save_name_prefix+f'_{i}_face_gt.obj', include_color=True)


    def latent_to_voxel(self, sdf_latent, face_latents):
        if sdf_latent is not None:
            sdf_latent = sdf_latent[None] # 1, C, N, N, N
            sdf_latent = sdf_latent * self.solid_latent_std + self.solid_latent_mean
            with torch.no_grad():
                sdf_voxel = self.sdf_model.quantize_decode(sdf_latent) # 1, 1, N, N, N
            sdf_voxel = sdf_voxel[0,0]
        else:
            sdf_voxel = None

        if face_latents is not None:
            face_latents = face_latents * self.face_latent_std + self.face_latent_mean
            with torch.no_grad():
                face_voxel = self.face_model.quantize_decode(face_latents) # M, 1, N, N, N
            face_voxel = face_voxel[:,0]
        else:
            face_voxel = None
        
        return sdf_voxel, face_voxel

    def render_mesh(self, voxel, filename, phase='sdf'):
        if filename is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        voxel = voxel.cpu().numpy()
        if phase == 'sdf':
            vertices, triangles = mcubes.marching_cubes(voxel, 0)
            mcubes.export_obj(vertices, triangles, filename)
        elif phase == 'face':
            points = np.where(voxel < 0.02)
            points = np.array(points).T
            pointcloud = trimesh.points.PointCloud(points)
            if filename is None:
                return pointcloud
            # save
            pointcloud.export(filename)
        else:
            raise ValueError(f'phase {phase} not supported')

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

