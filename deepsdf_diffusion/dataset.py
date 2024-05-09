import glob
import torch
import numpy as np
import os

class SDFDataset(torch.utils.data.Dataset):
    def __init__(self, data_config):
        self.data_config = data_config

        # load code_idx
        face_model_path = data_config['faces_model_path']
        face_code_idx = np.load(os.path.join(face_model_path, 'code_idx.npy'), allow_pickle=True).tolist()
        solid_model_path = data_config['solid_model_path']
        solid_code_idx = np.load(os.path.join(solid_model_path, 'code_idx.npy'), allow_pickle=True).tolist()

        # load code book
        face_codebook_path = glob.glob(os.path.join(face_model_path, 'checkpoints', 'last-*'))[0]
        self.face_codebook = \
            torch.load(face_codebook_path, map_location='cpu')['state_dict']['latent_code.weight']
        solid_codebook_path = glob.glob(os.path.join(solid_model_path, 'checkpoints', 'last-*'))[0]
        self.solid_codebook = \
            torch.load(solid_codebook_path, map_location='cpu')['state_dict']['latent_code.weight']
        
        # prepare data
        self.data_list = []
        for solid_key in solid_code_idx.keys():
            solid_idx = solid_code_idx[solid_key]
            solid_path = solid_key[:-2]
            face_idx_list = []
            for face_i in range(50):
                face_key = solid_path + '_' + str(face_i)
                if face_key not in face_code_idx:
                    break
                face_idx = face_code_idx[face_key]
                face_idx_list.append(face_idx)
            if len(face_idx_list) == 0:
                continue
            self.data_list.append((solid_idx, face_idx_list, solid_path))

        self.face_mean = self.face_codebook.mean()
        self.face_std = self.face_codebook.std()
        self.solid_mean = self.solid_codebook.mean()
        self.solid_std = self.solid_codebook.std()
        return

    def __len__(self):
        return len(self.data_list) 
    
    def __getitem__(self, idx):
        solid_idx, face_idx_list, solid_path = self.data_list[idx]
        solid_latent = self.solid_codebook[solid_idx] # ch
        face_latents = self.face_codebook[face_idx_list] # num_faces x ch

        faces_num = self.data_config['faces_num']
        if face_latents.shape[0] > faces_num:
            return self.__getitem__(torch.randint(0, len(self.data_list), (1,)).item())
        # shuffle face_latents
        face_latents = face_latents[torch.randperm(face_latents.shape[0])]

        if face_latents.shape[0] < faces_num:
            repeat_time = np.floor(faces_num / face_latents.shape[0]).astype(int)
            sep = faces_num - face_latents.shape[0] * repeat_time
            a = torch.cat([face_latents[:sep], ] * (repeat_time+1), 0)
            b = torch.cat([face_latents[sep:], ] * repeat_time, 0)
            face_latents = torch.cat([a, b], 0)
        
        return {'solid_latent': solid_latent, 'face_latents': face_latents, 
                'solid_path': solid_path, 
                'face_mean': self.face_mean, 'face_std': self.face_std,
                'solid_mean': self.solid_mean, 'solid_std': self.solid_std}


if __name__ == "__main__":
    import yaml
    with open('deepsdf_diffusion/configs/train_face_solid.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_dataset = SDFDataset(config['data_params']['train'])
    for idx, data in enumerate(train_dataset):
        print(idx)
    
    print('done')


