import open3d as o3d
import os
import pickle
import glob
import numpy as np

root_path = 'Data/deepcad_subset'
split = 'train'

valid_data_list = []
all_data_list = []
for objname in os.listdir(os.path.join(root_path, split)):
    obj_path = os.path.join(root_path, split, objname)
    solid_path_list = glob.glob(os.path.join(obj_path, 'solid_*.pkl'))
    for solid_path in solid_path_list:
        all_data_list.append(solid_path)
        stl_path = solid_path.replace('pkl', 'stl')
        solid_mesh = o3d.io.read_triangle_mesh(stl_path)
        solid_mesh.remove_duplicated_vertices()
        solid_mesh.remove_duplicated_triangles()
        bbox = solid_mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        extent_max = np.max(extent)
        extent_min = np.min(extent)
        if extent_max < extent_min * 15:
            valid_data_list.append(solid_path)

with open(f'{split}_valid_data_list.pkl', 'wb') as f:
    pickle.dump(valid_data_list, f)
