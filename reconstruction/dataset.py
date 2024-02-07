import glob
import torch
import pickle
import numpy as np
import os

class VoxelDataset(torch.utils.data.Dataset):
    def __init__(self, data_config, split):
        self.data_config = data_config
        if split == 'train':
            self.data_list = pickle.load(open(data_config['train_data_pkl_path'], 'rb'))
        elif split == 'val':
            self.data_list = pickle.load(open(data_config['val_data_pkl_path'], 'rb'))
    
    def __len__(self):
        return len(self.data_list) 
    
    def __getitem__(self, idx):
        pkl_path = self.data_list[idx]
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
        except:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        sdf_voxel = data['voxel_sdf'] # N,N,N
        face_dist = data['face_bounded_distance_field'] # N,N,N,M
        
        sdf_voxel = torch.from_numpy(sdf_voxel).float()
        face_dist = torch.from_numpy(face_dist).float()
        sdf_voxel = torch.clamp(sdf_voxel, -0.95, 0.95)
        face_dist = torch.clamp(face_dist, -0.95, 0.95)

        sdf_voxel = sdf_voxel.unsqueeze(0)
        face_dist = face_dist.permute(3, 0, 1, 2)
        face_dist = face_dist.unsqueeze(1)        
        return {'sdf': sdf_voxel, 'face_dist': face_dist,
                'face_num': face_dist.shape[0],
                'filename': "_".join(pkl_path.split('/')[-2:]).split('.')[0]}


if __name__ == "__main__":
    import yaml
    with open('vqvae/configs/vqvae_voxel_face_dist.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_dataset = VoxelDataset(config['data_params'], 'val')
    for data in train_dataset:
        print(data.shape)

