import glob
import torch
import pickle
import numpy as np
import OpenEXR
import Imath
import array
import os

class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, data_config, split):
        self.data_config = data_config
        if split == 'train':
            data_path = data_config['train_data_pkl_path']
            filter_path = data_config['train_filter_data_path']
        else:
            data_path = data_config['val_data_pkl_path']
            filter_path = data_config['val_filter_data_path']
        if os.path.isfile(data_path):
            self.data_list = pickle.load(open(data_path, 'rb'))
        else:
            self.data_list = glob.glob(os.path.join(data_path, '*.pkl'))
            self.data_list = sorted(self.data_list)
        
        #self.data_list = [self.data_list[4]]
        
        # filter
        if filter_path is not None:
            filter_list = pickle.load(open(filter_path, 'rb'))
            filter_list = [x.split('/')[-2]+'_'+x.split('/')[-1].split('.')[0] for x in filter_list]
            new_data_list = []
            for data in self.data_list:
                if 'lightning_logs' in data:
                    data_name = data.split('/')[-1].split('.')[0]
                else:
                    data_name = data.split('/')[-2]+'_'+data.split('/')[-1].split('.')[0]
                if data_name not in filter_list:
                    new_data_list.append(data)
                else:
                    print(f'Filter out {data_name}')
            self.data_list = new_data_list

        # cach data
        self.cache = {}
    
    def __len__(self):
        if self.data_config['debug']:
            return 400000
        return len(self.data_list) 
    
    def __getitem__(self, idx):
        if self.data_config['debug']:
            idx = idx % self.data_config['debug_size']
        idx = idx % len(self.data_list)
        pkl_path = self.data_list[idx]
        try:
            if idx in self.cache:
                data = self.cache[idx]['data']
            else:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                if len(self.cache) > self.data_config['cache_size']:
                    self.cache.popitem()
                self.cache[idx] = {'data': data}
        except:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        solid_voxel = data['voxel_sdf'] # 8,N,N,N
        face_voxel = data['face_bounded_distance_field'] # 8,N,N,N, M
        face_voxel = np.transpose(face_voxel, (4, 0, 1, 2, 3)) # M, 8, N,N,N
        num_faces = face_voxel.shape[0]
        if num_faces > self.data_config['max_faces']:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        
        cond_face_num = torch.randint(0, num_faces, (1,)).item()
        if self.data_config['face_shuffle']:
            face_voxel = face_voxel[torch.randperm(num_faces)]
        
        generate_face_num = num_faces - cond_face_num
        
        solid_voxel = torch.from_numpy(solid_voxel).float()[None]
        face_voxel = torch.from_numpy(face_voxel).float()

        cond_face_voxel = face_voxel[:cond_face_num]
        generate_face_voxel = face_voxel[cond_face_num:]

        # pad face to max number of faces
        max_faces = self.data_config['max_faces']
        generate_max_faces = max_faces - cond_face_num
        if generate_face_num < generate_max_faces:
            repeat_time = np.floor(generate_max_faces / generate_face_num).astype(int)
            sep = generate_max_faces - generate_face_num * repeat_time
            a = torch.cat([generate_face_voxel[:sep], ] * (repeat_time+1), 0)
            b = torch.cat([generate_face_voxel[sep:], ] * repeat_time, 0)
            generate_face_voxel = torch.cat([a, b], 0)
        
        face_voxel = torch.cat([cond_face_voxel, generate_face_voxel], 0)

        return {'solid_voxel': solid_voxel, 'face_voxel': face_voxel, 
                'cond_num': cond_face_num,
                'filename': pkl_path.split('/')[-1].split('.')[0]}


def read_exr(exr_path, height, width):
    file = OpenEXR.InputFile(exr_path)
    depth_arr = array.array('f', file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)))
    depth = np.array(depth_arr).reshape((height, width))
    depth[depth < 0] = 0
    depth[depth > 10] = 0
    return depth


def depth2pcd(depth, intrinsics, pose):
    inv_K = np.linalg.inv(intrinsics)
    y,x = np.where(depth > 0)
    depth_value = depth[y,x]
    point_cam = depth_value * inv_K.dot(np.stack([x, y, np.ones_like(x)], 0))

    R = pose[:, :3]
    t = pose[:, 3]

    point_world = np.linalg.inv(R).dot(point_cam - t[:, None])
    
    return point_world.T


class PartialPCLatentDataset(LatentDataset):
    def __init__(self, data_config, split):
        super().__init__(data_config, split)

    def __getitem__(self, idx):
        if self.data_config['debug']:
            idx = idx % self.data_config['debug_size']
        idx = idx % len(self.data_list)
        pkl_path = self.data_list[idx]
        try:
            if idx in self.cache:
                data = self.cache[idx]['data']
                meta = self.cache[idx]['meta']
                points_list = self.cache[idx]['points_list']
            else:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
            
                data_name = pkl_path.split('/')[-1].split('_')[0]
                solid_name = pkl_path.split('/')[-1].split('.')[0].split('_')[-1]
                solid_name = 'solid_' + solid_name
                render_path = os.path.join(self.pointcloud_path, 
                                               data_name, 'render', solid_name)
                meta = pickle.load(open(os.path.join(render_path, 'meta.pkl'), 'rb'))
                K = meta[0]
                poses = meta[1]
                # load depths
                all_depth_files = glob.glob(os.path.join(render_path, '*.exr'))
                all_depth_files = sorted(all_depth_files)
                points_list = []
                for depth_idx, depth_file in enumerate(all_depth_files):
                    depth = read_exr(depth_file, 256, 256)
                    pose = poses[depth_idx] # 3x4
                    points = depth2pcd(depth, K, pose)
                    points_list.append(points)
                
                if len(self.cache) > self.data_config['cache_size']:
                    self.cache.popitem()
                self.cache[idx] = {'data': data, 'meta': meta, 'points_list': points_list}
                
            # make pointcloud
            num_views = torch.randint(self.data_config['num_min_views'], 
                                      self.data_config['num_max_views']+1, (1,)).item()
            if num_views > 0:
                point_idxs = torch.randperm(len(points_list))[:num_views]
            
                all_points = []
                for point_idx in point_idxs:
                    all_points.append(points_list[point_idx])
                all_points = np.concatenate(all_points, 0)
    
                all_points[:,1] = -all_points[:,1]
                pointcloud = all_points[:,[0,2,1]]
            else:
                pointcloud = np.zeros((0,3))

        except:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        solid_voxel = data['voxel_sdf'] # 8,N,N,N
        face_voxel = data['face_bounded_distance_field'] # 8,N,N,N, M
        face_voxel = np.transpose(face_voxel, (4, 0, 1, 2, 3)) # M, 8, N,N,N
        num_faces = face_voxel.shape[0]
        if num_faces > self.data_config['max_faces']:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        
        solid_voxel = torch.from_numpy(solid_voxel).float()[None]
        face_voxel = torch.from_numpy(face_voxel).float()
        if self.data_config['face_shuffle']:
            face_voxel = face_voxel[torch.randperm(num_faces)]

        # pad face to max number of faces
        max_faces = self.data_config['max_faces']
        if num_faces < max_faces:
            repeat_time = np.floor(max_faces / num_faces).astype(int)
            sep = max_faces - num_faces * repeat_time
            a = torch.cat([face_voxel[:sep], ] * (repeat_time+1), 0)
            b = torch.cat([face_voxel[sep:], ] * repeat_time, 0)
            face_voxel = torch.cat([a, b], 0)

        # voxelize pointcloud [-1, 1] -> [0, 63]
        pointcloud = (pointcloud + 1) / 2 * (64-1)
        pointcloud = pointcloud.astype(int)
        pc_voxel = np.zeros((64, 64, 64))
        pc_voxel[pointcloud[:,0], pointcloud[:,1], pointcloud[:,2]] = 1
        pc_voxel = pc_voxel * 2 - 1
        pc_voxel = pc_voxel.transpose(1,0,2)
        pc_voxel = torch.from_numpy(pc_voxel).float()[None]

        return {'solid_voxel': solid_voxel, 'face_voxel': face_voxel, 
                'pc_voxel': pc_voxel, 
                'face_num': num_faces, 'is_latent': 1,
                'filename': pkl_path.split('/')[-1].split('.')[0]}


class LatentDataset_temp(torch.utils.data.Dataset):
    def __init__(self, data_config, split):
        self.data_config = data_config
        if split == 'train':
            if os.path.isfile(data_config['train_data_pkl_path']):
                self.data_list = pickle.load(open(data_config['train_data_pkl_path'], 'rb'))
            else:
                self.data_list = glob.glob(os.path.join(data_config['train_data_pkl_path'], '*.pkl'))
                self.data_list = sorted(self.data_list)
        elif split == 'val':
            if os.path.isfile(data_config['val_data_pkl_path']):
                self.data_list = pickle.load(open(data_config['val_data_pkl_path'], 'rb'))
            else:
                self.data_list = glob.glob(os.path.join(data_config['val_data_pkl_path'], '*.pkl'))
                self.data_list = sorted(self.data_list)
        self.pad_latent = np.load(data_config['fake_latent_path'])
    
    def __len__(self):
        if self.data_config['debug']:
            return 150000
        return len(self.data_list) 
    
    def __getitem__(self, idx):
        if self.data_config['debug']:
            idx = idx % self.data_config['debug_size']
        pkl_path = self.data_list[idx]
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
        except:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        sdf_voxel = data['voxel_sdf'] # 8,N,N,N
        face_voxel = data['face_bounded_distance_field'] # 8,N,N,N, M
        face_voxel = np.transpose(face_voxel, (4, 0, 1, 2, 3)) # M, 8, N,N,N
        num_faces = face_voxel.shape[0]

        sdf_voxel = torch.from_numpy(sdf_voxel).float()[None]
        face_voxel = torch.from_numpy(face_voxel).float()
        if self.data_config['face_shuffle']:
            face_voxel = face_voxel[torch.randperm(num_faces)]
        
        face_voxel = face_voxel[:self.data_config['max_faces']]
        num_faces = face_voxel.shape[0]

        # pad face to max number of faces
        max_faces = self.data_config['max_faces']
        if num_faces < max_faces:
            if self.data_config['pad_fake_latent']:
                pad = torch.from_numpy(self.pad_latent)
                pad = pad[None]
                pad = pad.repeat(max_faces - num_faces, 1, 1, 1, 1)
                face_voxel = torch.cat([face_voxel, pad], 0)
            else:
                repeat_time = np.floor(max_faces / num_faces).astype(int)
                sep = max_faces - num_faces * repeat_time
                a = torch.cat([face_voxel[:sep], ] * (repeat_time+1), 0)
                b = torch.cat([face_voxel[sep:], ] * repeat_time, 0)
                face_voxel = torch.cat([a, b], 0)
        return {'sdf_voxel': sdf_voxel, 'face_voxel': face_voxel, 
                'face_num': num_faces, 'is_latent': 1,
                'filename': pkl_path.split('/')[-1].split('.')[0]}

if __name__ == "__main__":
    import yaml
    with open('pointcloud_diffusion/configs/train_point_partial_v2.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_dataset = LatentDataset(config['data_params'],'val')
    for _ in range(3):
        for idx, data in enumerate(train_dataset):
            print(idx)
    
        print('done')


