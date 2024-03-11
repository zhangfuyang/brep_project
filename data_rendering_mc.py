import pickle
import numpy as np
import glob
import torch
import mcubes
import os
import trimesh
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', 
                    type=str, 
                    default='cond_diffusion_transformer/logs/cond_solid_debug/lightning_logs/version_0/')
parser.add_argument('--folder_name', 
                    type=str, default='test')
parser.add_argument('--save_root', default='')
parser.add_argument('--apply_nms', action='store_true', default=False)
args = parser.parse_args()

data_root = args.data_root
folder_name = args.folder_name # pkl
save_root = args.save_root if args.save_root != '' else os.path.join(data_root, 'test_render')
grid_reso = 64

class B_edges:
    def __init__(self) -> None:
        self.vertices = np.array([[],[],[]]).T
        self.edges = np.array([[],[]], dtype=int).T
    
    def add_vertex(self, vertex):
        ### return vertex idx
        # check if the vertex is already in the list
        if len(self.vertices) == 0:
            self.vertices = np.array([vertex])
            return 0
        dist = np.linalg.norm(self.vertices - vertex, axis=1)
        if np.min(dist) > 1e-6:
            self.vertices = np.vstack([self.vertices, vertex])
            return len(self.vertices) - 1
        else:
            return np.argmin(dist)
    
    def add_edge(self, edge):
        ### return edge idx
        # check if the edge is already in the list
        if len(self.edges) == 0:
            self.edges = np.array([edge])
            return 0
        dist = np.linalg.norm(self.edges - edge, axis=1)
        if np.min(dist) > 1e-6:
            self.edges = np.vstack([self.edges, edge])
            return len(self.edges) - 1
        else:
            return np.argmin(dist)
    
    def export_vertices(self, filename):
        pc = trimesh.points.PointCloud(self.vertices)
        pc.colors = np.ones((len(self.vertices), 4)) * trimesh.visual.random_color()
        pc.export(filename, include_color=True)
    
    def visualize(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        #ax.scatter(self.vertices[:,0], self.vertices[:,1], self.vertices[:,2])
        # random a color
        color = np.random.rand(3)
        for edge in self.edges:
            ax.plot(self.vertices[edge, 0], 
                    self.vertices[edge, 1], 
                    self.vertices[edge, 2], color=color)

def nms(f_bdf):
    # f_bdf: (64, 64, 64, N)
    # NMS
    def similarity(a, b, threshold=0.03):
        A = np.abs(a) < threshold
        B = np.abs(b) < threshold
        return np.sum(A & B) / (np.sum(A | B) + 1e-8)

    valid_idx = [0]
    for i in range(1, f_bdf.shape[-1]):
        is_valid = True
        for j in valid_idx:
            if similarity(f_bdf[..., i], f_bdf[..., j]) > 0.9:
                is_valid = False
                break
        if is_valid:
            valid_idx.append(i)
    return f_bdf[..., valid_idx]
                
def clean_bdf(f_bdf, threshold=0.05):
    valid_idx = []
    for i in range(f_bdf.shape[-1]):
        if np.sum(np.abs(f_bdf[..., i]) < threshold) > 0:
            valid_idx.append(i)
    return f_bdf[..., valid_idx]


data_path_list = glob.glob(os.path.join(data_root, f'{folder_name}/*.pkl'))
for data_path in data_path_list:
    print(data_path)
    data_name = data_path.split('/')[-1].split('.')[0]
    save_base = os.path.join(save_root, data_name)
    os.makedirs(save_base, exist_ok=True)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    v_sdf = data['voxel_sdf']
    f_bdf = data['face_bounded_distance_field']
    if isinstance(v_sdf, torch.Tensor):
        v_sdf = v_sdf.cpu().numpy()
    if isinstance(f_bdf, torch.Tensor):
        f_bdf = f_bdf.cpu().numpy()
    if args.apply_nms:
        f_bdf = clean_bdf(f_bdf)
        f_bdf = nms(f_bdf)

    vertices, triangles = mcubes.marching_cubes(v_sdf, 0)

    mcubes.export_obj(vertices, triangles, f'{save_base}/mc_mesh.obj')
    solid_pc = trimesh.points.PointCloud(vertices)
    solid_pc.export(f'{save_base}/mc_vertice.obj', include_color=True)

    all_intersection_points = []

    f_dbf_interpolator = RegularGridInterpolator(
        (np.arange(grid_reso), np.arange(grid_reso), np.arange(grid_reso)), f_bdf, 
        bounds_error=False, fill_value=0)

    interpolated_f_bdf = f_dbf_interpolator(vertices) # v, num_faces
    vertices_face_id = interpolated_f_bdf.argmin(-1)

    # save vertice points per face
    for i in range(f_bdf.shape[-1]):
        v = vertices[vertices_face_id == i]
        if v.shape[0] == 0:
            continue
        pc = trimesh.points.PointCloud(v)
        pc.colors = np.ones((v.shape[0], 4)) * trimesh.visual.random_color()
        pc.export(f'{save_base}/face_{i}.obj')

    boundary_dict = {}
    for tri in triangles:
        three_face_id = vertices_face_id[tri]
        if len(np.unique(three_face_id)) == 1:
            continue
        elif len(np.unique(three_face_id)) == 2:
            ids = np.unique(three_face_id)
            group_a, group_b = tri[three_face_id == ids[0]], tri[three_face_id == ids[1]]
            if len(group_a) == 1:
                two_points = group_b
                one_point = group_a
            else:
                two_points = group_a
                one_point = group_b

            # get the intersection points
            point1 = (vertices[two_points[0]] + vertices[one_point[0]]) / 2
            point2 = (vertices[two_points[1]] + vertices[one_point[0]]) / 2

            ids = np.sort(ids)
            if (ids[0], ids[1]) not in boundary_dict:
                boundary_dict[(ids[0], ids[1])] = B_edges()
                edges = boundary_dict[(ids[0], ids[1])]
            else:
                edges = boundary_dict[(ids[0], ids[1])]
            idx1 = edges.add_vertex(point1)
            idx2 = edges.add_vertex(point2)
            edges.add_edge([idx1, idx2])
        else:
            center_point = np.mean(vertices[tri], axis=0)
            for i in range(3):
                two_ids = three_face_id[[i, (i+1)%3]]
                point = (vertices[tri[i]] + vertices[tri[(i+1)%3]]) / 2

                ids = np.sort(two_ids)
                if (ids[0], ids[1]) not in boundary_dict:
                    boundary_dict[(ids[0], ids[1])] = B_edges()
                    edges = boundary_dict[(ids[0], ids[1])]
                else:
                    edges = boundary_dict[(ids[0], ids[1])]
                idx1 = edges.add_vertex(center_point)
                idx2 = edges.add_vertex(point)
                edges.add_edge([idx1, idx2])

    # export the boundary vertices
    for k, v in boundary_dict.items():
        v.export_vertices(f'{save_base}/boundary_{k[0]}_{k[1]}.obj')

    
    # check face only
    for face_i in range(f_bdf.shape[-1]):
        face = f_bdf[..., face_i]
        points = np.where(face < 0.1)
        points = np.array(points).T
        pointcloud = trimesh.points.PointCloud(points)
        # save
        filename = f'{save_base}/face_only_{face_i}.obj'
        pointcloud.export(filename)

    

