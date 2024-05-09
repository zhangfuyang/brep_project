import glob
import torch
import pickle
import numpy as np
import os
import glob

class VoxelDataset(torch.utils.data.Dataset):
    def __init__(self, data_config):
        self.data_config = data_config
        self.data_list = glob.glob(os.path.join(data_config['data_path'], '*', 'solid*.npz'))
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_path = self.data_list[idx]
        try:
            data = np.load(data_path)
            voxel = data[self.data_config['data_key']]
        except:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        if voxel.ndim == 4:
            random_idx = torch.randint(0, voxel.shape[-1], (1,)).item()
            voxel = voxel[..., random_idx]
        
        x = torch.from_numpy(voxel).float()
        x = torch.clamp(x, -self.data_config['clip_value'], self.data_config['clip_value'])
        x = x[None] # 1, N,N,N
        return x

if __name__ == "__main__":
    import yaml
    with open('vae/configs/train_solid.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_dataset = VoxelDataset(config['data_params']['train'])
    for data in train_dataset:
        print(data.shape)
