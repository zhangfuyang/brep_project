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
        voxel = data[self.data_config['data_key']]
        if voxel.ndim == 4:
            random_idx = torch.randint(0, voxel.shape[-1], (1,)).item()
            voxel = voxel[..., random_idx]
        
        x = torch.from_numpy(voxel).float()
        x = torch.clamp(x, -0.3, 0.3)
        x = x * 3 # [-0.3, 0.3] -> [-0.9, 0.9]
        x = x[None] # 1, N,N,N
        return x

if __name__ == "__main__":
    import yaml
    with open('vqvae/configs/vqvae_voxel_sdf.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_dataset = VoxelDataset(config['data_params'], 'val')
    for data in train_dataset:
        print(data.shape)
