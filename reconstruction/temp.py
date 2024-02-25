import pickle
import numpy as np
import os
import glob

data_dir = 'reconstruction/logs/latent_dim_4_val/lightning_logs/version_0/pkl'

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

voxel_max = 0
face_max = 0
for _ in range(500):
    i = np.random.randint(voxel_sdf.shape[-1])
    j = np.random.randint(voxel_sdf.shape[-1])
    xi = voxel_sdf[...,i]
    xj = voxel_sdf[...,j]
    voxel_norm = np.linalg.norm(xi-xj)
    voxel_max = voxel_norm if voxel_norm > voxel_max else voxel_max

    i = np.random.randint(face_udf.shape[-1])
    j = np.random.randint(face_udf.shape[-1])
    xi = face_udf[...,i]
    xj = face_udf[...,j]
    face_norm = np.linalg.norm(xi-xj)
    face_max = face_norm if face_norm > face_max else face_max

print(voxel_max, face_max)
    
    
