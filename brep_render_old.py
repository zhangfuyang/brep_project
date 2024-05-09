import pickle
import numpy as np
import torch
import mcubes
import os
import trimesh
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import argparse
import scipy
from numpy.polynomial.polynomial import Polynomial


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', 
                    type=str, 
                    default='pointcloud_diffusion/logs/debug_new_model/lightning_logs/version_9/test/0004/gt/raw.pkl')
                    #default='sdf_diffusion/logs/latent_full_solid_cond/lightning_logs/version_4/test/0015.pkl')
parser.add_argument('--save_root', default='temp/render_17')
parser.add_argument('--apply_nms', action='store_true', default=False)
parser.add_argument('--vis_each_face', action='store_true', default=False)
parser.add_argument('--vis_face_all', action='store_true', default=False)
parser.add_argument('--vis_face_only', action='store_true', default=False)
parser.add_argument('--vis_each_boundary', action='store_true', default=False)
parser.add_argument('--vis_boundary_all', action='store_true', default=False)
parser.add_argument('--vis_boundary_png', action='store_true', default=False)
parser.add_argument('--export_pkl', action='store_true', default=False)

args = parser.parse_args()

base_color = np.array(
    [[255,   0,  0, 255],  # Red
    [  0, 255,   0, 255],  # Green
    [  0,   0, 255, 255],  # Blue
    [255, 255,   0, 255],  # Yellow
    [  0, 255, 255, 255],  # Cyan
    [255,   0, 255, 255],  # Magenta
    [255, 165,   0, 255],  # Orange
    [128,   0, 128, 255],  # Purple
    [255, 192, 203, 255],  # Pink
    [128, 128, 128, 255],  # Gray
    [210, 245, 60, 255], # Lime
    [170, 110, 40, 255], # Brown
    [128, 0, 0, 255], # Maroon
    [0, 128, 128, 255], # Teal
    [0, 0, 128, 255], # Navy
    ], dtype=np.uint8
)

class Brep:
    def __init__(self) -> None:
        self.faces = []
        self.boundaries = set()

    def add_face(self, face):
        self.faces.append(face)
        face.parent_brep = self
    
    def add_boundary(self, boundary):
        self.boundaries.add(boundary)
    
    def export(self, split_boundary=True):
        boundaries_list = []
        boundaries_instance_list = []
        faces_list = []
        for boundary in self.boundaries:
            curr_b = boundary.export(split_into_groups=split_boundary)
            boundaries_list.extend(curr_b)
            boundaries_instance_list.extend([boundary for i in range(len(curr_b))])
        for face in self.faces:
            face_mesh = face.export()
            boundary_idx = []
            for i in range(len(boundaries_instance_list)):
                if boundaries_instance_list[i] in face.boundaries:
                    boundary_idx.append(i)
            face_mesh['boundaries'] = boundary_idx
            faces_list.append(face_mesh)
        
        return {'boundaries': boundaries_list, 'faces': faces_list}

    def visualize(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0,64)
        ax.set_ylim(0,64)
        ax.set_zlim(0,64)
        for bd in self.boundaries:
            for func in bd.parametric_functions:
                px, py, pz, t = func
                x = px(t)
                y = py(t)
                z = pz(t)
                ax.plot(x, y, z)
        
        plt.savefig('cccc.png', dpi=300)

class Face:
    def __init__(self, id_, mesh) -> None:
        self.id = id_
        self.boundaries = []
        self.mesh = mesh
        # add color
        mesh.visual.vertex_colors = np.ones((len(mesh.vertices), 4)) * base_color[id_ % len(base_color)]
        self.parent_brep = None
    
    def add_boundary(self, boundary):
        self.boundaries.append(boundary)
        boundary.parent_faces.append(self)
    
    def fit_face(self):
        # fit the face
        # ax^2 + by^2 + cz^2 + dxy + exz + fyz + gx + hy + iz + j = 0
        vertices = self.mesh.vertices
        target = 0

        # constraints: must pass the start and end points of each boundary
        must_pass_points = []
        for bd in self.boundaries:
            bd_v = bd.vertices
            for bd_group in bd.groups:
                v_start = bd_v[bd_group[0]]
                v_end = bd_v[bd_group[-1]]
                must_pass_points.append(v_start)
                must_pass_points.append(v_end)
        must_pass_points = np.array(must_pass_points)


        def func(coeff, data):
            # data: (N, 3)
            a, b, c, d, e, f, g, h, i, j = coeff
            x = data[:, 0]
            y = data[:, 1]
            z = data[:, 2]
            return a*x**2 + b*y**2 + c*z**2 + d*x*y + e*x*z + f*y*z + g*x + h*y + i*z + j
        
        def objective(coeff, data, target):
            pred = func(coeff, data)
            return np.sum((pred - target) ** 2)
        
        def constraint(coeff):
            pred = func(coeff, must_pass_points)
            return np.sum((pred - target)**2)
        
        opt = scipy.optimize.minimize(
            objective, x0=np.random.rand(10), 
            args=(vertices, target), 
            constraints={'type': 'eq', 'fun': constraint})
        
        self.opt_coeff = opt.x

        return self.opt_coeff
    
    def plot_parametric_face(self, save_path):
        # plot the parametric face
        # ax^2 + by^2 + cz^2 + dxy + exz + fyz + gx + hy + iz + j = 0
        def func(x, y, z):
            # data: (N, 3)
            a, b, c, d, e, f, g, h, i, j = self.opt_coeff
            return a*x**2 + b*y**2 + c*z**2 + d*x*y + e*x*z + f*y*z + g*x + h*y + i*z + j

        mesh_point = self.mesh.vertices # (N, 3)

        ax = plt.figure().add_subplot(111, projection='3d')
        # plot mesh point first
        ax.scatter(mesh_point[:, 0], mesh_point[:, 1], mesh_point[:, 2], c='r')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        def plot_implicit():
            #xmax, ymax, zmax = np.max(mesh_point, axis=0) + 2
            #xmin, ymin, zmin = np.min(mesh_point, axis=0) - 2
            xmax = ymax = zmax = np.max(mesh_point) + 2
            xmin = ymin = zmin = np.min(mesh_point) - 2
            X = np.linspace(xmin, xmax, 100) 
            Y = np.linspace(ymin, ymax, 100)
            Z = np.linspace(zmin, zmax, 100) 

            XY = np.meshgrid(X,Y)
            XZ = np.meshgrid(X,Z)
            YZ = np.meshgrid(Y,Z)

            for z in Z: # plot contours in the XY plane
                x, y = XY
                pred = func(x,y,z)
                cset = ax.contour(x, y, pred+z, [z], zdir='z')

            for y in Y: # plot contours in the XZ plane
                x,z = XZ
                pred = func(x,y,z)
                cset = ax.contour(x, pred+y, z, [y], zdir='y')

            for x in X: # plot contours in the YZ plane
                y,z = YZ
                pred = func(x,y,z)
                cset = ax.contour(pred+x, y, z, [x], zdir='x')

            ax.set_zlim3d(zmin,zmax)
            ax.set_xlim3d(xmin,xmax)
            ax.set_ylim3d(ymin,ymax)

        plot_implicit()
        plt.savefig(save_path)

    def export(self):
        return {'vertices': self.mesh.vertices, 'triangle': self.mesh.triangles}

    def visualize(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(0,64)
            ax.set_ylim(0,64)
            ax.set_zlim(0,64)
        ax.scatter(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], self.mesh.vertices[:, 2])
        return ax
        

class Boundary:
    def __init__(self) -> None:
        self.vertices = np.array([[],[],[]]).T
        self.vertices_type = np.array([], dtype=int)
        self.connections = np.array([[],[]], dtype=int).T
        self.groups = None
        self.parent_faces = []
        self.parametric_functions = []

    def add_vertex(self, vertex, v_type=0):
        if len(self.vertices) == 0:
            self.vertices = np.array([vertex])
            self.vertices_type = np.array([v_type])
            return 0
        dist = np.linalg.norm(self.vertices - vertex, axis=1)
        if np.min(dist) > 1e-6:
            self.vertices = np.vstack([self.vertices, vertex])
            self.vertices_type = np.hstack([self.vertices_type, v_type])
            return len(self.vertices) - 1
        else:
            return np.argmin(dist)

    def add_connection(self, connection):
        ### return edge idx
        # check if the edge is already in the list
        connection = sorted(connection)
        if len(self.connections) == 0:
            self.connections = np.array([connection])
            return 0
        dist = np.linalg.norm(self.connections - connection, axis=1)
        if np.min(dist) > 1e-6:
            self.connections = np.vstack([self.connections, connection])
            return len(self.connections) - 1
        else:
            return np.argmin(dist)
    
    def split_into_edges(self):
        v_belongs = [-1 for _ in range(len(self.vertices))]
        groups = []
        for connection_i in range(len(self.connections)):
            v0, v1 = self.connections[connection_i]
            if v_belongs[v0] == -1 and v_belongs[v1] == -1:
                groups.append([v0, v1])
                v_belongs[v0] = len(groups) - 1
                v_belongs[v1] = len(groups) - 1
            elif v_belongs[v0] == -1:
                v_belongs[v0] = v_belongs[v1]
                group = groups[v_belongs[v1]]
                v1_loc = group.index(v1)
                assert v1_loc == 0 or v1_loc == len(group) - 1
                if v1_loc == 0:
                    group.insert(0, v0)
                else:
                    group.append(v0)
            elif v_belongs[v1] == -1:
                v_belongs[v1] = v_belongs[v0]
                group = groups[v_belongs[v0]]
                v0_loc = group.index(v0)
                assert v0_loc == 0 or v0_loc == len(group) - 1
                if v0_loc == 0:
                    group.insert(0, v1)
                else:
                    group.append(v1)
            else:
                if v_belongs[v0] != v_belongs[v1]:
                    group0 = groups[v_belongs[v0]]
                    group1 = groups[v_belongs[v1]]
                    v0_loc = group0.index(v0)
                    v1_loc = group1.index(v1)
                    assert v0_loc == 0 or v0_loc == len(group0) - 1
                    assert v1_loc == 0 or v1_loc == len(group1) - 1
                    if v0_loc == 0:
                        group0[:] = group0[::-1]
                    if v1_loc == len(group1) - 1:
                        group1[:] = group1[::-1]
                    # now v0 is the last element of group0, 
                    # v1 is the first element of group1
                    group0.extend(group1)
                    removed_group_idx = v_belongs[v1]
                    groups.pop(removed_group_idx)
                    for v_idx in group1:
                        v_belongs[v_idx] = v_belongs[v0]
                    for i in range(len(v_belongs)):
                        if v_belongs[i] > removed_group_idx:
                            v_belongs[i] -= 1
        
        # filter some bad groups
        for g in groups:
            if self.vertices_type[g[0]] != 1 or self.vertices_type[g[-1]] != 1:
                groups.remove(g)
                continue
            if len(g) < 15:
                groups.remove(g)
                continue
        self.groups = groups

        return len(self.groups)
                    
    def export(self, split_into_groups=True):
        if split_into_groups:
            groups = []
            if self.groups is None:
                self.split_into_edges()
            for i in range(len(self.groups)):
                group = self.groups[i]
                vertices = self.vertices[group]
                groups.append(vertices)
            return groups
        else:
            return [self.vertices]

    def visualize_vertices(self, filename, split_into_groups=False):
        if split_into_groups:
            assert filename.endswith('.obj') is False
            if self.groups is None:
                self.split_into_edges()
            for i in range(len(self.groups)):
                group = self.groups[i]
                vertices = self.vertices[group]
                pc = trimesh.points.PointCloud(vertices)
                pc.colors = np.ones((len(vertices), 4)) * trimesh.visual.random_color()
                pc.export(f'{filename}_{i}.obj', include_color=True)
            return
        pc = trimesh.points.PointCloud(self.vertices)
        pc.colors = np.ones((len(self.vertices), 4)) * trimesh.visual.random_color()
        pc.export(filename, include_color=True)
    
    def compute_parametric_functions(self):
        # compute the parametric functions for each group
        for group in self.groups:
            self.parametric_functions.append(self.compute_parametric_function(group))
        
    def compute_parametric_function(self, group):
        # compute the parametric function for the group
        # group: list of vertex idx
        x = self.vertices[group][:,0]
        y = self.vertices[group][:,1]
        z = self.vertices[group][:,2]
        weights = np.ones(len(x))
        weights[0] = weights[-1] = 1000

        t = np.zeros(x.shape[0])
        for i in range(1, t.shape[0]):
            t[i] = t[i-1] + np.linalg.norm(self.vertices[group[i]] - self.vertices[group[i-1]])

        # Fit polynomials to the data
        # check to use deg=1 or 2
        
        last_error = 10000
        for deg in [1, 2]:
            px = Polynomial.fit(t, x, deg=deg, w=weights)
            py = Polynomial.fit(t, y, deg=deg, w=weights)
            pz = Polynomial.fit(t, z, deg=deg, w=weights)
            x_fit, y_fit, z_fit = px(t), py(t), pz(t)
            error = np.mean((x - x_fit)**2 + (y - y_fit)**2 + (z - z_fit)**2)
            if error > last_error / 10 or error < 5:
                break
            else:
                last_error = error

        t = np.linspace(t[0], t[-1], int(t[-1] * 10))

        return [px, py, pz, t]

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
            if similarity(f_bdf[..., i], f_bdf[..., j]) > 0.3:
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

data_path = args.data_path
save_base = args.save_root
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

grid_reso = f_bdf.shape[0]
f_dbf_interpolator = RegularGridInterpolator(
    (np.arange(grid_reso), np.arange(grid_reso), np.arange(grid_reso)), f_bdf, 
    bounds_error=False, fill_value=0)
v_sdf_interpolator = RegularGridInterpolator(
    (np.arange(grid_reso), np.arange(grid_reso), np.arange(grid_reso)), v_sdf, 
    bounds_error=False, fill_value=0)

interpolated_f_bdf = f_dbf_interpolator(vertices) # v, num_faces
vertices_face_id = interpolated_f_bdf.argmin(-1)

mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
mesh.visual.vertex_colors = base_color[vertices_face_id % len(base_color)]
mesh.export(f'{save_base}/mc_mesh_color.obj', include_color=True)

brep = Brep()
triangle_face_id = vertices_face_id[triangles] # M, 3
for face_idx in range(f_bdf.shape[-1]):
    triangle_belong_face = (triangle_face_id == face_idx).all(1)
    if np.sum(triangle_belong_face) == 0:
        continue
    # get valid triangle idx [True, False, True, True] -> [0, 2, 3]
    triangle_idx = np.arange(len(triangles))[triangle_belong_face]
    face_mesh = mesh.submesh([triangle_idx], append=True)
    face = Face(face_idx, face_mesh)
    brep.add_face(face)
    if args.vis_each_face:
        face.mesh.export(f'{save_base}/face_{face_idx}.obj', include_color=True)

if args.vis_face_all:
    faces = brep.faces
    all_face = None
    for face_idx, face in enumerate(faces):
        all_face = face.mesh if all_face is None else all_face + face.mesh
    all_face.export(f'{save_base}/face_all.obj', include_color=True)

# find the boundary
boundary_dict = {}
for tri_idx, tri in enumerate(triangles):
    three_vertice_id = vertices_face_id[tri]
    if len(np.unique(three_vertice_id)) == 1:
        continue
    elif len(np.unique(three_vertice_id)) == 2:
        ids = np.unique(three_vertice_id)
        group_a, group_b = tri[three_vertice_id == ids[0]], tri[three_vertice_id == ids[1]]
        if len(group_a) == 1:
            two_points = group_b
            one_point = group_a
        else:
            two_points = group_a
            one_point = group_b

        # get the intersection points & find two triangles of the intersection points
        point1 = (vertices[two_points[0]] + vertices[one_point[0]]) / 2
        point2 = (vertices[two_points[1]] + vertices[one_point[0]]) / 2

        ids = np.sort(ids)
        if (ids[0], ids[1]) not in boundary_dict:
            boundary_dict[(ids[0], ids[1])] = Boundary()
            boundary = boundary_dict[(ids[0], ids[1])]
        else:
            boundary = boundary_dict[(ids[0], ids[1])]
        idx1 = boundary.add_vertex(point1)
        idx2 = boundary.add_vertex(point2)
        boundary.add_connection([idx1, idx2])
    else:
        center_point = np.mean(vertices[tri], axis=0)
        for i in range(3):
            two_ids = three_vertice_id[[i, (i+1)%3]]
            point = (vertices[tri[i]] + vertices[tri[(i+1)%3]]) / 2

            ids = np.sort(two_ids)
            if (ids[0], ids[1]) not in boundary_dict:
                boundary_dict[(ids[0], ids[1])] = Boundary()
                boundary = boundary_dict[(ids[0], ids[1])]
            else:
                boundary = boundary_dict[(ids[0], ids[1])]
            idx1 = boundary.add_vertex(center_point, v_type=1)
            idx2 = boundary.add_vertex(point)
            boundary.add_connection([idx1, idx2])

# add boundary to each face and brep
for k, v in boundary_dict.items():
    num_edges = v.split_into_edges() # divide the boundary into groups
    if num_edges == 0:
        raise ValueError('No valid edges')
    face_a = faces[k[0]]
    face_b = faces[k[1]]
    face_a.add_boundary(v)
    face_b.add_boundary(v)
    brep.add_boundary(v)
    v.compute_parametric_functions()

brep.visualize()

ax = plt.figure().add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(0,64)
ax.set_ylim(0,64)
ax.set_zlim(0,64)
for face in brep.faces:
    face.visualize(ax)
    plt.savefig(f'cccc_face.png')


#for face in faces:
#    face.fit_face()
#    face.plot_parametric_face(f'{save_base}/parametric_face_{face.id}.png')

if args.export_pkl:
    # export brep
    brep_data = brep.export()
    with open(f'{save_base}/brep.pkl', 'wb') as f:
        pickle.dump(brep_data, f)

if args.vis_each_boundary:
    # export the boundary vertices
    for k, v in boundary_dict.items():
        v.visualize_vertices(f'{save_base}/boundary_{k[0]}_{k[1]}', split_into_groups=True)

if args.vis_face_only:
    for face_i in range(f_bdf.shape[-1]):
        face = f_bdf[..., face_i]
        points = np.where(face < 0.1)
        points = np.array(points).T
        pointcloud = trimesh.points.PointCloud(points)
        # save
        filename = f'{save_base}/face_only_{face_i}.obj'
        pointcloud.export(filename)


