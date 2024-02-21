import pickle
import numpy as np
import os
import glob

data_dir = 'reconstruction/logs/vq_latent/lightning_logs/version_0/pkl'

voxel_all = []
face_all = []
for pkl_name in glob.glob(os.path.join(data_dir, '*.pkl')):
    with open(pkl_name, 'rb') as f:
        data = pickle.load(f)
    voxel_sdf = data['voxel_sdf'][..., None]
    face_udf = data['face_bounded_distance_field']

    voxel_all.append(voxel_sdf)
    face_all.append(face_udf)

voxel_sdf = np.concatenate(voxel_all, axis=-1)
face_udf = np.concatenate(face_all, axis=-1)

voxel_mean, voxel_std = np.mean(voxel_sdf), np.std(voxel_sdf)
face_mean, face_std = np.mean(face_udf), np.std(face_udf)

print(f'voxel mean: {voxel_mean}, voxel std: {voxel_std}')
print(f'face mean: {face_mean}, face std: {face_std}')

