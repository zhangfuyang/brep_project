import os
from tqdm import tqdm
import open3d as o3d
import numpy as np
import glob
import open3d as o3d
import argparse

def scale_to_unit_cube(mesh, centroid, max_size, unit=1.0):
    mesh.translate(-centroid)
    mesh.scale(2 * unit / max_size, center=(0, 0, 0))

def sample_points_on_mesh(mesh, num_points):
    points = mesh.sample_points_uniformly(number_of_points=num_points)
    points = np.asarray(points.points)
    #half_size = points.shape[0] // 2
    #points[:half_size] += np.random.normal(0, 0.1, points[:half_size].shape)
    #points[half_size:] += np.random.normal(0, 0.01, points[half_size:].shape)
    
    return points

def uniform_sample_points_in_unit_sphere(num_points, unit=1.0):
    points = np.random.uniform(-unit, unit, (num_points, 3))
    #points = points[np.linalg.norm(points, axis=-1) < 1]
    #points_available = points.shape[0]
    #if points_available < num_points:
    #    result = np.zeros((num_points, 3))
    #    result[:points_available] = points
    #    result[points_available:] = uniform_sample_points_in_unit_sphere(num_points-points_available)
    #    return result
    return points[:num_points]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--group_num', type=int, default=1)
    parser.add_argument('--group_id', type=int, default=0)
    parser.add_argument('--sampling_points', type=int, default=10000)
    parser.add_argument('--save_root', type=str, default='Data/deepcad_subset')
    parser.add_argument('--max_faces', type=int, default=25)
    args = parser.parse_args()

    root_path = 'Data/deepcad_subset'
    split = args.split # train, val, test

    data_ids = os.listdir(os.path.join(root_path, split))
    data_ids.sort()
    data_ids = data_ids[args.group_id::args.group_num]
    os.makedirs(os.path.join(args.save_root, split), exist_ok=True)

    for data_id in tqdm(data_ids):
        #try:
            data_dir = os.path.join(root_path, split, data_id)
            # Load the step file
            save_dir = os.path.join(args.save_root, split, data_id)
            for solid_stl_path in glob.glob(os.path.join(save_dir, 'solid_*.stl')):
                solid_mesh = o3d.io.read_triangle_mesh(solid_stl_path)
                solid_mesh.remove_duplicated_vertices()
                solid_mesh.remove_duplicated_triangles()
                # check if the mesh is watertight
                if solid_mesh.is_watertight() is False:
                    break

                # normalise the mesh
                bbox = solid_mesh.get_axis_aligned_bounding_box()
                centroid = bbox.get_center()
                max_size = np.max(bbox.get_extent()) * 1.2
                scale_to_unit_cube(solid_mesh, centroid, max_size, unit=1.0)

                # save normalized mesh
                save_path = solid_stl_path.replace('.stl', '_norm.obj')
                o3d.io.write_triangle_mesh(save_path, solid_mesh)

                # sample points on the solid surface
                solid_i = int(solid_stl_path.split('_')[-1].split('.')[0])
                if os.path.exists(os.path.join(save_dir, f'solid_{solid_i}_surf_point.npy')):
                    continue
                num_points = args.sampling_points
                surface_points = sample_points_on_mesh(solid_mesh, num_points)
                points = surface_points
                points = points.astype(np.float32)

                data = {}
                data['solid_surface_point'] = points
                np.save(os.path.join(save_dir, f'solid_{solid_i}_surf_point.npy'), data)
        #except:
        #    continue
            

