import glob
import torch
import numpy as np
import os

class SDFDataset(torch.utils.data.Dataset):
    def __init__(self, data_config):
        self.data_config = data_config
        data_root = data_config['data_path']
        
        #data_paths = glob.glob(os.path.join(data_root, '*', '*.npy'))
        data_paths = np.load('deepsdf/logs/faces/lightning_logs/version_0/temp_path.npy', allow_pickle=True).tolist()
        data_paths = sorted(data_paths)

        # count how many data we have
        data_key = data_config['data_key']
        if data_key == 'face_udf':
            self.data_list = []
            for data_path in data_paths:
                data = np.load(data_path, allow_pickle=True).tolist()
                sdf_ = data[data_key]
                for i in range(sdf_.shape[0]):
                    self.data_list.append((data_path, i))
        else:
            self.data_list = [(data_path, 0) for data_path in data_paths]
        
        self.data_id = [i for i in range(len(self.data_list))] # used for data code

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        #idx = idx % 500
        data_path = self.data_list[idx]
        data_path, data_idx = data_path
        data = np.load(data_path, allow_pickle=True).tolist()
        if self.data_config['data_key'] == 'face_udf':
            p_dist = data['face_udf'][data_idx]
        else:
            p_dist = data['solid_sdf']

        points = p_dist[:, :3] # Nx3
        dist = p_dist[:, 3] # N

        points = torch.from_numpy(points).float()
        dist = torch.from_numpy(dist).float()
        dist = torch.clamp(dist, self.data_config['margin_beta'][0], self.data_config['margin_beta'][1])

        num_points = self.data_config['num_points_per_data']
        rand_idx = torch.randint(0, points.shape[0], (num_points,))
        points = points[rand_idx]
        dist = dist[rand_idx]

        data_id = self.data_id[idx]

        return points, dist, data_id


if __name__ == "__main__":
    import yaml
    with open('deepsdf/configs/sdf_face.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_dataset = SDFDataset(config['data_params']['train'])
    for idx, data in enumerate(train_dataset):
        print(idx)

