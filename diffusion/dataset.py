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
            pad = torch.zeros((max_faces - num_faces, *face_voxel.shape[1:]))
            face_voxel = torch.cat([face_voxel, pad], 0)
        return {'sdf_voxel': sdf_voxel, 'face_voxel': face_voxel, 'face_num': num_faces}


if __name__ == "__main__":
    import yaml
    with open('vqvae/configs/vqvae_voxel_face_dist.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_dataset = VoxelDataset(config['data_params'], 'val')
    for data in train_dataset:
        print(data.shape)


