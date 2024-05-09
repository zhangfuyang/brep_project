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
from scipy.interpolate import RegularGridInterpolator

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
            self.solid_latent_mean = data['solid_mean']
            self.solid_latent_std = data['solid_std']
            if self.config['visual_type'] == 'sanity_check':
                self.face_latent_mean = self.solid_latent_mean
                self.face_latent_std = self.solid_latent_std
            else:
                self.face_latent_mean = data['face_mean']
                self.face_latent_std = data['face_std']
        else:
            self.face_latent_mean = 0
            self.face_latent_std = 1
            self.solid_latent_mean = 0
            self.solid_latent_std = 1

    @torch.no_grad()
    def preprocess(self, batch):
        solid_voxel = batch['solid_voxel'] # bs, 1, N, N, N or bs, 1, 8, N, N, N
        face_voxel = batch['face_voxel'] # bs, M, N, N, N or bs, M, 8, N, N, N

        # get latent
        if solid_voxel.dim() == 5:
            # voxel
            with torch.no_grad():
                solid_latent = self.sdf_model.encode(solid_voxel)[:,None] # bs, 1, C, N, N, N

                self.latent_dim = solid_latent.shape[2]

                bs = face_voxel.shape[0]
                face_voxel = face_voxel.reshape(-1, *face_voxel.shape[2:])[:,None]
                face_latent = self.face_model.encode(face_voxel) # bs*M, C, N, N, N
                face_latent = face_latent.reshape(bs, -1, *face_latent.shape[1:])
                # bs, M, C, N, N, N
        else:
            self.latent_dim = solid_voxel.shape[2]
            solid_latent = solid_voxel
            face_latent = face_voxel
        
        # normalize
        solid_latent = (solid_latent - self.solid_latent_mean) / self.solid_latent_std
        face_latent = (face_latent - self.face_latent_mean) / self.face_latent_std

        bs = solid_latent.shape[0]
        latent = torch.cat([solid_latent, face_latent], 1) # bs, 1+M, C, N, N, N

        return latent
    
    def training_step(self, batch, batch_idx):
        x = self.preprocess(batch)
        face_latent = x[:,1:]
        solid_latent = x[:,:1]
        
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (x.shape[0],), device=x.device).long()
        noise_face = torch.randn_like(face_latent)
        noise_solid = torch.randn_like(solid_latent)
        face_latent_noisy = self.scheduler.add_noise(face_latent, noise_face, t)
        solid_latent_noisy = self.scheduler.add_noise(solid_latent, noise_solid, t)
        
        model_output_face, model_output_solid = self.diffusion_model(
                            face_latent_noisy, solid_latent_noisy, t)

        solid_target = solid_latent if self.config['prediction_type'] == 'sample' else noise_solid
        face_target = face_latent if self.config['prediction_type'] == 'sample' else noise_face
        
        target = torch.cat([solid_target, face_target], 1)
        model_output = torch.cat([model_output_solid, model_output_face], 1)
        #target = face_target
        #model_output = model_output_face
        if self.config['loss_type'] == 'l2':
            sd_loss = torch.nn.functional.mse_loss(model_output, target)
        elif self.config['loss_type'] == 'l1':
            sd_loss = torch.nn.functional.l1_loss(model_output, target)

        self.log('train_loss', sd_loss, rank_zero_only=True, prog_bar=True)

        loss = sd_loss 

        return loss
    
    def on_train_epoch_end(self) -> None:
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", cur_lr, prog_bar=True)
    
    def nms_latent(self, latent):
        # latent: (M, ...)
        # NMS
        def similarity(a, b, threshold=0.03):
            return (a-b).abs().sum()
        
        valid_idx = []
        for i in range(0, latent.shape[0]):
            is_valid = True
            for j in valid_idx:
                if similarity(latent[i], latent[j]) < 0.03:
                    is_valid = False
                    break
            if is_valid:
                valid_idx.append(i)
        
        return latent[valid_idx]

    def nms(self, f_bdf, must_keep=None):
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

        if must_keep is None:
            must_keep = []
        valid_idx = []
        for i in range(0, f_bdf.shape[0]):
            is_valid = True
            if i in must_keep:
                valid_idx.append(i)
                continue
            for j in valid_idx:
                if similarity(f_bdf[i], f_bdf[j]) > 0.2:
                    is_valid = False
                    break
                coverage_ij = coverage(f_bdf[i], f_bdf[j])
                coverage_ji = coverage(f_bdf[j], f_bdf[i])
                if coverage_ij > 0.7 or coverage_ji > 0.7:
                    if coverage_ij > coverage_ji:
                        is_valid = False
                        break
                    else:
                        if j in must_keep:
                            is_valid = False
                            break
                        else:
                            valid_idx.remove(j)
            if is_valid:
                valid_idx.append(i)
        return f_bdf[valid_idx]

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        x = self.preprocess(batch)
        face_latent = x[:,1:]
        solid_latent = x[:,:1]
        z_faces = torch.randn_like(face_latent)
        z_solid = torch.randn_like(solid_latent)

        self.scheduler.set_timesteps(self.config['diffusion_steps'])
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            z_faces_input = z_faces 
            z_solid_input = z_solid 
            timestep = torch.cat([t.unsqueeze(0)]*z_faces_input.shape[0], 0)
            model_output_face, model_output_solid = self.diffusion_model(
                                z_faces_input, z_solid_input, timestep)
            z_solid = self.scheduler.step(model_output_solid, t, z_solid).prev_sample
            z_faces = self.scheduler.step(model_output_face, t, z_faces).prev_sample

        if self.trainer.is_global_zero:
            if batch_idx == 0: 
                for i in range(min(400, x.shape[0])):
                    if self.config['visual_type'] == 'vanilla':
                        os.makedirs(os.path.join(self.logger.log_dir, 'image'), exist_ok=True)
                        save_name_prefix = os.path.join(self.logger.log_dir, 'image', f'{self.global_step}_{i}_pred')
                        solid_voxel, face_voxel = self.latent_to_voxel(z_solid[i][0], z_faces[i])
                        solid_mesh = self.render_mesh(solid_voxel, phase='sdf')
                        solid_mesh.export(save_name_prefix+'_solid.obj')
                        face_meshes = None
                        for face_idx in range(5):
                            face_mesh = self.render_mesh(face_voxel[face_idx], phase='face')
                            if face_mesh.vertices.shape[0] == 0:
                                continue
                            face_mesh.visual.vertex_colors = base_color[face_idx]
                            if face_meshes is None:
                                face_meshes = face_mesh
                            else:
                                face_meshes += face_mesh
                        face_meshes.export(save_name_prefix+'_face.obj', include_color=True)
                        
                        save_name_prefix = os.path.join(self.logger.log_dir, 'image', f'{self.global_step}_{i}_gt')
                        solid_voxel, face_voxel = self.latent_to_voxel(solid_latent[i][0], face_latent[i])
                        solid_mesh = self.render_mesh(solid_voxel, phase='sdf')
                        solid_mesh.export(save_name_prefix+'_solid.obj')
                        face_meshes = None
                        for face_idx in range(3):
                            face_mesh = self.render_mesh(face_voxel[face_idx], phase='face')
                            face_mesh.visual.vertex_colors = base_color[face_idx]
                            if face_meshes is None:
                                face_meshes = face_mesh
                            else:
                                face_meshes += face_mesh
                        face_meshes.export(save_name_prefix+'_face.obj', include_color=True)
                    elif self.config['visual_type'] == 'brep':
                        save_name = os.path.join(self.logger.log_dir, 'val', f'{self.global_step}', f'{i}_pred', 'raw.pkl')
                        self.save_result(z_solid[i][0], z_faces[i], save_name)
                        save_name = os.path.join(self.logger.log_dir, 'val', f'{self.global_step}', f'{i}_gt', 'raw.pkl')
                        self.save_result(solid_latent[i][0], face_latent[i], save_name)
                        # make a file with filename
                        gt_filename = batch['filename'][i]
                        with open(os.path.join(os.path.dirname(save_name), f'{gt_filename}'), 'w') as f:
                            f.write(gt_filename)
                    elif self.config['visual_type'] == 'sanity_check':
                        os.makedirs(os.path.join(self.logger.log_dir, 'image'), exist_ok=True)
                        save_name_prefix = os.path.join(self.logger.log_dir, 'image', f'{self.global_step}_{i}_pred')
                        solid_voxel, _ = self.latent_to_voxel(z_solid[i][0], None)
                        solid_mesh = self.render_mesh(solid_voxel, phase='sdf')
                        solid_mesh.export(save_name_prefix+'_solid.obj')
                        face_voxel, _ = self.latent_to_voxel(z_faces[i][0], None)
                        face_mesh = self.render_mesh(face_voxel, phase='sdf')
                        face_mesh.export(save_name_prefix+'_face.obj')
                        
                        save_name_prefix = os.path.join(self.logger.log_dir, 'image', f'{self.global_step}_{i}_gt')
                        solid_voxel, _ = self.latent_to_voxel(solid_latent[i][0], None)
                        solid_mesh = self.render_mesh(solid_voxel, phase='sdf')
                        solid_mesh.export(save_name_prefix+'_solid.obj')
                        face_voxel, _ = self.latent_to_voxel(face_latent[i][0], None)
                        face_mesh = self.render_mesh(face_voxel, phase='sdf')
                        face_mesh.export(save_name_prefix+'_face.obj')


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

    def render_mesh(self, voxel, filename=None, phase='sdf'):
        if filename is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        voxel = voxel.cpu().numpy()
        if phase == 'sdf':
            vertices, triangles = mcubes.marching_cubes(voxel, 0)
            if filename is None:
                return trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
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

    def render_mesh_2(self, solid_voxel, face_voxel, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        solid_voxel = solid_voxel.cpu().numpy()
        face_voxel = face_voxel.cpu().numpy()
        face_voxel = face_voxel.transpose(1, 2, 3, 0) # N, N, N, M
        vertices, triangles = mcubes.marching_cubes(solid_voxel, 0)
        grid_reso = face_voxel.shape[0]
        f_dbf_interpolator = RegularGridInterpolator(
            (np.arange(grid_reso), np.arange(grid_reso), np.arange(grid_reso)), face_voxel, 
            bounds_error=False, fill_value=0)
        interpolated_f_bdf = f_dbf_interpolator(vertices) # v, num_faces
        vertices_face_id = interpolated_f_bdf.argmin(-1)
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
        if vertices_face_id.shape[0] != 0:
            mesh.visual.vertex_colors = base_color[vertices_face_id % len(base_color)]
        mesh.export(filename, include_color=True)

    def configure_optimizers(self):
        for module_name in self.config['freeze_modules']:
            module = self.diffusion_model.__dict__['_modules'][module_name]
            for param in module.parameters():
                param.requires_grad = False
        p = filter(lambda p: p.requires_grad, self.diffusion_model.parameters())
        optimizer = torch.optim.Adam(
            p, lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['lr_decay'])
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def save_result(self, solid_latent, face_latent, save_name):
        solid_voxel, face_voxels = self.latent_to_voxel(solid_latent, face_latent)
        #face_voxels = self.nms(face_voxels)

        #self.render_mesh_2(solid_voxel, face_voxels, os.path.dirname(save_name)+f'/mc.obj')
        solid_voxel = solid_voxel.cpu().numpy()
        face_voxels = face_voxels.cpu().numpy()

        face_voxels = face_voxels.transpose(1, 2, 3, 0)

        data = {}
        data['voxel_sdf'] = solid_voxel
        data['face_bounded_distance_field'] = face_voxels
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        with open(save_name, 'wb') as f:
            pickle.dump(data, f)
        command = f'python brep_render.py --data_path {save_name} --save_root {os.path.join(os.path.dirname(save_name), "processed")} ' \
                    f'--apply_nms --vis_each_face --vis_face_all --vis_face_only --vis_each_boundary'
        print(command)
        os.system(command)

        # save raw
        os.makedirs(os.path.join(os.path.dirname(save_name), 'raw'), exist_ok=True)
        for face_i in range(face_voxels.shape[-1]):
            face_v = face_voxels[:,:,:,face_i]
            points = np.where(face_v < 0.02)
            points = np.array(points).T
            pointcloud = trimesh.points.PointCloud(points)
            # save
            pointcloud.export(os.path.join(os.path.dirname(save_name), 'raw', f'face_{face_i}_only.obj'))

    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        x = self.preprocess(batch)

        self.scheduler.set_timesteps(self.config['diffusion_steps'])
        timesteps = self.scheduler.timesteps

        face_cond_num = batch['cond_num']
        face_latent = x[:,1:]
        solid_latent = x[:,:1]

        for round_i in range(3):
            z_faces = torch.randn_like(face_latent)
            z_solid = torch.randn_like(solid_latent)
            cond_mask = torch.zeros_like(face_latent[:,:,:1]) # bs, M, 1, N, N, N
            for batch_i in range(x.shape[0]):
                cond_mask[batch_i, :face_cond_num[batch_i]] = 1
                z_faces[batch_i, :face_cond_num[batch_i]] = face_latent[batch_i, :face_cond_num[batch_i]]

            for i, t in enumerate(timesteps):
                z_faces_input = z_faces 
                z_solid_input = z_solid 
                timestep = torch.cat([t.unsqueeze(0)]*z_faces_input.shape[0], 0)
                model_output_face, model_output_solid = self.diffusion_model(
                                    z_faces_input, z_solid_input, cond_mask, timestep)
                z_solid = self.scheduler.step(model_output_solid, t, z_solid).prev_sample
                z_faces = self.scheduler.step(model_output_face, t, z_faces).prev_sample
                for batch_i in range(x.shape[0]):
                    z_faces[batch_i, :face_cond_num[batch_i]] = face_latent[batch_i, :face_cond_num[batch_i]]

            if self.trainer.is_global_zero:
                for i in range(min(20, x.shape[0])):
                    cur_test_idx = self.test_idx + i
                    save_name = os.path.join(self.logger.log_dir, 'test', f'{cur_test_idx:04d}', f'round_{round_i}', 'raw.pkl')
                    self.save_result(z_solid[i][0], z_faces[i], face_cond_num[i].item(), save_name)
                    save_name = os.path.join(self.logger.log_dir, 'test', f'{cur_test_idx:04d}', f'gt', 'raw.pkl')
                    self.save_result(x[i][0], x[i][1:], face_cond_num[i].item(), save_name)

        self.test_idx += x.shape[0]
    

