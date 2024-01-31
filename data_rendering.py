import pickle
import numpy as np
import glob
import mcubes
import os
import trimesh
from scipy.interpolate import RegularGridInterpolator

face_dist_threshold = 0.04
grid_reso = 64

data_path = glob.glob('Data/deepcad_subset/train/*/*.pkl')[100]

save_root = 'debug_train_2'
os.makedirs(save_root, exist_ok=True)
os.system(f'cp -r {"/".join(data_path.split("/")[:-1])} {save_root}/debug_original')

with open(data_path, 'rb') as f:
    data = pickle.load(f)

v_sdf = data['voxel_sdf']
f_bdf = data['face_bounded_distance_field']

vertices, triangles = mcubes.marching_cubes(v_sdf, 0)

mcubes.export_obj(vertices, triangles, f'{save_root}/mc_mesh.obj')
solid_pc = trimesh.points.PointCloud(vertices)
solid_pc.export(f'{save_root}/mc_vertice.obj', include_color=True)

all_intersection_points = []

f_dbf_interpolator = RegularGridInterpolator(
    (np.arange(grid_reso), np.arange(grid_reso), np.arange(grid_reso)), f_bdf, 
    bounds_error=False, fill_value=0)

interpolated_f_bdf = f_dbf_interpolator(vertices)

boundary_points = []
edge_points = []
corner_points = []
per_edge_points = {}
for vertice_idx in range(len(vertices)):
    face_distances = interpolated_f_bdf[vertice_idx]
    flag = face_distances < face_dist_threshold
    if flag.sum() == 1:
        boundary_points.append(vertices[vertice_idx])
    elif flag.sum() == 2:
        edge_points.append(vertices[vertice_idx])
        face_ids = np.where(flag)[0]
        face_ids.sort()
        if (face_ids[0], face_ids[1]) in per_edge_points:
            per_edge_points[(face_ids[0], face_ids[1])].append(vertices[vertice_idx])
        else:
            per_edge_points[(face_ids[0], face_ids[1])] = [vertices[vertice_idx]]
    elif flag.sum() >= 3:
        corner_points.append(vertices[vertice_idx])

if len(boundary_points) > 0:
    boundary_pc_trimesh = trimesh.points.PointCloud(np.array(boundary_points))
    boundary_pc_trimesh.colors = np.ones((len(boundary_points), 4)) * [0, 1, 0, 1]
    boundary_pc_trimesh.export(f'{save_root}/boundary_points.obj', include_color=True)

if len(edge_points) > 0:
    edge_pc_trimesh = trimesh.points.PointCloud(np.array(edge_points))
    edge_pc_trimesh.colors = np.ones((len(edge_points), 4)) * [0, 1, 0, 1]
    edge_pc_trimesh.export(f'{save_root}/edge_points.obj', include_color=True)

if len(corner_points) > 0:
    corner_pc_trimesh = trimesh.points.PointCloud(np.array(corner_points))
    corner_pc_trimesh.colors = np.ones((len(corner_points), 4)) * [0, 1, 0, 1]
    corner_pc_trimesh.export(f'{save_root}/corner_points.obj', include_color=True)

for key_ in per_edge_points:
    if len(per_edge_points[key_]) > 0:
        per_edge_pc_trimesh = trimesh.points.PointCloud(np.array(per_edge_points[key_]))
        per_edge_pc_trimesh.colors = np.ones((len(per_edge_points[key_]), 4)) * trimesh.visual.random_color()
        per_edge_pc_trimesh.export(f'{save_root}/per_edge_points_{key_}.obj', include_color=True)

    

