import glob
import torch
import pickle
import numpy as np
import os

class VoxelDatasetOld(torch.utils.data.Dataset):
    def __init__(self, data_config, split):
        self.data_config = data_config
        if split == 'train':
            self.data_list = pickle.load(open(data_config['train_data_pkl_path'], 'rb'))
        elif split == 'val':
            self.data_list = pickle.load(open(data_config['val_data_pkl_path'], 'rb'))
    
    def __len__(self):
        return len(self.data_list) 
    
    def __getitem__(self, idx):
        if self.data_config['debug']:
            idx = idx % 1
        pkl_path = self.data_list[idx]
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
        except:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        sdf_voxel = data['voxel_sdf'] # N,N,N
        face_voxel = data['face_bounded_distance_field'] # N,N,N, M
        face_voxel = np.transpose(face_voxel, (3, 0, 1, 2)) # M, N,N,N
        num_faces = face_voxel.shape[0]
        if num_faces > self.data_config['max_faces']:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        
        sdf_voxel = torch.from_numpy(sdf_voxel).float()[None]
        face_voxel = torch.from_numpy(face_voxel).float()
        sdf_voxel = torch.clamp(sdf_voxel, -0.95, 0.95)
        face_voxel = torch.clamp(face_voxel, -0.95, 0.95)
        if self.data_config['face_shuffle']:
            face_voxel = face_voxel[torch.randperm(num_faces)]

        # pad face to max number of faces
        max_faces = self.data_config['max_faces']
        if num_faces < max_faces:
            pad = torch.ones((max_faces - num_faces, *face_voxel.shape[1:]))
            face_voxel = torch.cat([face_voxel, pad], 0)
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
        elif split == 'val':
            if os.path.isfile(data_config['val_data_pkl_path']):
                self.data_list = pickle.load(open(data_config['val_data_pkl_path'], 'rb'))
            else:
                self.data_list = glob.glob(os.path.join(data_config['val_data_pkl_path'], '*.pkl'))
        self.pad_latent = np.load(data_config['fake_latent_path'])
    
    def __len__(self):
        if self.data_config['debug']:
            return 10000
        return len(self.data_list) 
    
    def __getitem__(self, idx):
        if self.data_config['debug']:
            idx = idx % 5
        pkl_path = self.data_list[idx]
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
        except:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        sdf_voxel = data['voxel_sdf'] # 8,N,N,N
        face_voxel = data['face_bounded_distance_field'] # 8,N,N,N, M
        face_voxel = np.transpose(face_voxel, (4, 0, 1, 2, 3)) # M, 8, N,N,N
        # select random number of conditions
        m = face_voxel.shape[0]
        if m > self.data_config['max_faces']:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        num_cond = torch.randint(0, m, (1,)).item()
        permutation = torch.randperm(m)
        cond_idx = permutation[:num_cond]
        target_idx = permutation[num_cond:]

        face_condition = face_voxel[cond_idx] if len(cond_idx) != 1 else face_voxel[cond_idx][None] # num_cond, 8, N,N,N
        # random select one target exclude the condition
        if len(target_idx) == 0:
            face_target = self.pad_latent # 8, N,N,N
        else:
            face_target = face_voxel[target_idx[0]] # 8, N,N,N

        sdf_voxel = torch.from_numpy(sdf_voxel).float()[None]
        face_cond_voxel = torch.from_numpy(face_condition).float()

        # pad face to max number of faces
        max_faces = self.data_config['max_faces']+1
        if num_cond < max_faces:
            pad = torch.from_numpy(self.pad_latent)
            pad = pad[None]
            pad = pad.repeat(max_faces - num_cond, 1, 1, 1, 1)
            face_cond_voxel = torch.cat([face_cond_voxel, pad], 0)
        return {'sdf_voxel': sdf_voxel, 'face_cond_voxel': face_cond_voxel, 
                'face_target_voxel': face_target, 'is_latent': 1,
                'filename': pkl_path.split('/')[-1].split('.')[0]}

class TestDataset(LatentDataset):
    def __init__(self, data_config, split):
        super().__init__(data_config, split)
    
    def __len__(self):
        return 20
    
    def __getitem__(self, idx):
        if self.data_config['debug']:
            idx = idx % 5
        pkl_path = self.data_list[idx]
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
        except:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        sdf_voxel = data['voxel_sdf'][None] # 8,N,N,N

        face_cond_voxel = self.pad_latent[None]

        return {'sdf_voxel': sdf_voxel, 'face_cond_voxel': face_cond_voxel, 
                'is_latent': 1,
                'filename': pkl_path.split('/')[-1].split('.')[0]}


if __name__ == "__main__":
    import yaml
    with open('regress_cond_diffusion/configs/diffusion_vq_latent.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_dataset = LatentDataset(config['data_params'], 'train')
    for idx, data in enumerate(train_dataset):
        print(idx)
    
    print('done')


