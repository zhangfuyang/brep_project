import glob
import torch
import pickle
import numpy as np
import os

class VoxelDataset(torch.utils.data.Dataset):
    def __init__(self, data_config, split):
        self.data_config = data_config
        if split == 'train':
            self.data_list = glob.glob(data_config['train_data_pkl_path'])
            self.data_list = sorted(self.data_list)
        elif split == 'val':
            self.data_list = glob.glob(data_config['val_data_pkl_path'])
            self.data_list = sorted(self.data_list)
    
    def __len__(self):
        return len(self.data_list) 
    
    def __getitem__(self, idx):
        if self.data_config['debug']:
            idx = idx % 20
        pkl_path = self.data_list[idx]
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
        except:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        sdf_voxel = data['voxel_sdf'][None] # 1,N,N,N
        face_voxel = data['face_bounded_distance_field'] # N,N,N, M
        face_voxel = np.transpose(face_voxel, (3, 0, 1, 2)) # M, N,N,N
        face_voxel = face_voxel[:,None] # M, 1, N,N,N
        num_faces = face_voxel.shape[0]
        if num_faces > self.data_config['max_faces']:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        
        sdf_voxel = torch.from_numpy(sdf_voxel).float()[None]
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
        return {'sdf_voxel': sdf_voxel, 'face_voxel': face_voxel, 
                'face_num': num_faces, 'is_latent': 0,
                'filename': pkl_path.split('/')[-1].split('.')[0]}

class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, data_config, split):
        self.data_config = data_config
        if split == 'train':
            if os.path.isfile(data_config['train_data_pkl_path']):
                self.data_list = pickle.load(open(data_config['train_data_pkl_path'], 'rb'))
            else:
                self.data_list = glob.glob(os.path.join(data_config['train_data_pkl_path'], '*.pkl'))
        elif split == 'val':
            if os.path.isfile(data_config['val_data_pkl_path']):
                self.data_list = pickle.load(open(data_config['val_data_pkl_path'], 'rb'))
            else:
                self.data_list = glob.glob(os.path.join(data_config['val_data_pkl_path'], '*.pkl'))

        self.fake_data = np.load('pad_latent.npy')

        self.cache = {}
        if self.data_config['use_cache']:
            for idx in range(len(self.data_list)):
                pkl_path = self.data_list[idx]
                try:
                    with open(pkl_path, 'rb') as f:
                        data = pickle.load(f)
                except:
                    pass
                self.cache[idx] = data
        

    def __len__(self):
        return len(self.data_list) 
    
    def __getitem__(self, idx):
        if torch.randint(0, 100, (1,)).item() == 0 and self.data_config['data_key'] != 'voxel_sdf':
            voxel = self.fake_data
        else:
            if idx in self.cache:
                data = self.cache[idx]
            else:
                pkl_path = self.data_list[idx]
                try:
                    with open(pkl_path, 'rb') as f:
                        data = pickle.load(f)
                except:
                    return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
            voxel = data[self.data_config['data_key']] # 8,N,N,N or 8,N,N,N,M
            if voxel.ndim == 5:
                voxel_idx = torch.randint(0, voxel.shape[-1], (1,)).item()
                voxel = voxel[...,voxel_idx]
        
        return {'x': voxel}

class LatentDataset(torch.utils.data.Dataset):
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
            return 30000
        return len(self.data_list) * 10
    
    def __getitem__(self, idx):
        if self.data_config['debug']:
            idx = idx % self.data_config['debug_size']
        idx = idx % len(self.data_list)
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
        if num_faces > self.data_config['max_faces']:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        
        sdf_voxel = torch.from_numpy(sdf_voxel).float()[None]
        face_voxel = torch.from_numpy(face_voxel).float()
        if self.data_config['face_shuffle']:
            face_voxel = face_voxel[torch.randperm(num_faces)]

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

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_config, split):
        self.data_config = data_config
    
    def __len__(self):
        return 20
    
    def __getitem__(self, idx):
        sdf_voxel = torch.rand(1, 64,64,64)
        face_voxel = torch.rand(10, 64,64,64)
        face_num = 10
        return {'sdf_voxel': sdf_voxel, 'face_voxel': face_voxel, 'face_num': face_num}

class BBoxDataset(torch.utils.data.Dataset):
    def __init__(self, data_config, split):
        self.data_config = data_config
        if split == 'train':
            if os.path.isfile(data_config['train_data_pkl_path']):
                self.data_list = pickle.load(open(data_config['train_data_pkl_path'], 'rb'))
            else:
                self.data_list = glob.glob(os.path.join(data_config['train_data_pkl_path'], '*.npy'))
                self.data_list = sorted(self.data_list)
        elif split == 'val':
            if os.path.isfile(data_config['val_data_pkl_path']):
                self.data_list = pickle.load(open(data_config['val_data_pkl_path'], 'rb'))
            else:
                self.data_list = glob.glob(os.path.join(data_config['val_data_pkl_path'], '*.npy'))
                self.data_list = sorted(self.data_list)
    
    def __len__(self):
        if self.data_config['debug']:
            return 300000
        return len(self.data_list) 
    
    def __getitem__(self, idx):
        if self.data_config['debug']:
            idx = idx % self.data_config['debug_size'] + 1
        pkl_path = self.data_list[idx]
        try:
            with open(pkl_path, 'rb') as f:
                face_bbox = np.load(f)
        except:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        face_voxel = torch.from_numpy(face_bbox).float()[...,None,None,None]
        num_faces = face_voxel.shape[0]
        if num_faces != self.data_config['max_faces']:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        sdf_voxel = torch.zeros_like(face_voxel[:1])
        if self.data_config['face_shuffle']:
            face_voxel = face_voxel[torch.randperm(num_faces)]

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
        obj_path = pkl_path.replace('face_bbox', 'solid').replace('.npy', '.stl')
        return {'sdf_voxel': sdf_voxel, 'face_voxel': face_voxel, 
                'face_num': num_faces, 'is_latent': 1,
                'filename': obj_path}

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_config, split):
        self.data_config = data_config
    
    def __len__(self):
        return 20
    
    def __getitem__(self, idx):
        sdf_voxel = torch.rand(1, 64,64,64)
        face_voxel = torch.rand(10, 64,64,64)
        face_num = 10
        return {'sdf_voxel': sdf_voxel, 'face_voxel': face_voxel, 'face_num': face_num}

class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, data_config, split):
        self.data_config = data_config
        latent_size = self.data_config['latent_size']
        self.faces = torch.ones(1, 5, 16, latent_size, latent_size, latent_size).float()
        for i in range(self.faces.shape[1]):
            self.faces[:,i] = i / (self.faces.shape[1] - 1) * 2 - 1
        self.solid = torch.ones(1, 1, 16, latent_size, latent_size, latent_size).float()

    def __len__(self):
        if self.data_config['debug']:
            return 30000
        return self.faces.shape[0]
    
    def __getitem__(self, idx):
        if self.data_config['debug']:
            idx = idx % self.data_config['debug_size']
        sdf_voxel = self.solid[idx] # 1, C, N, N, N
        face_voxel = self.faces[idx] # M, C, N, N, N
        num_faces = face_voxel.shape[0]
        if num_faces > self.data_config['max_faces']:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        
        sdf_voxel = sdf_voxel.float()
        face_voxel = face_voxel.float()
        if self.data_config['face_shuffle']:
            face_voxel = face_voxel[torch.randperm(num_faces)]

        return {'sdf_voxel': sdf_voxel, 'face_voxel': face_voxel, 
                'face_num': num_faces, 'is_latent': 1,}

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

class LatentDataset_cond(torch.utils.data.Dataset):
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
    with open('sdf_diffusion/configs/train_bbox.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_dataset = BBoxDataset(config['data_params'], 'val')
    for idx, data in enumerate(train_dataset):
        print(idx)
    
    print('done')


