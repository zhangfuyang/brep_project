import os
from tqdm import tqdm
import open3d as o3d
from occwl.io import load_step, save_stl
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer
from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Wire, ShapeFix_Edge
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
import numpy as np
import glob
import open3d as o3d
import pickle
import argparse

def fix_wires(face, debug=False):
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        if debug:
            wire_checker = ShapeAnalysis_Wire(wire, face, 0.01)
            print(f"Check order 3d {wire_checker.CheckOrder()}")
            print(f"Check 3d gaps {wire_checker.CheckGaps3d()}")
            print(f"Check closed {wire_checker.CheckClosed()}")
            print(f"Check connected {wire_checker.CheckConnected()}")
        wire_fixer = ShapeFix_Wire(wire, face, 0.01)
        assert wire_fixer.IsReady()
        ok = wire_fixer.Perform()

def fix_face(face):
    fixer = ShapeFix_Face(face)
    fixer.SetPrecision(0.01)
    fixer.SetMaxTolerance(0.1)
    ok = fixer.Perform()
    # assert ok
    fixer.FixOrientation()
    face = fixer.Face()
    return face

def add_pcurves_to_edges(face):
    edge_fixer = ShapeFix_Edge()
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        wire_exp = WireExplorer(wire)
        for edge in wire_exp.ordered_edges():
            edge_fixer.FixAddPCurve(edge, face, False, 0.001)

def split_solid(solid):
    '''
    Split closed faces on the solid into halve
    '''
    solid = solid.split_all_closed_faces(num_splits=0)
    solid = solid.split_all_closed_edges(num_splits=0)
    split_faces = []
    for face in solid.faces():
        face_occ = face.topods_shape()
        # Do some fixing
        fix_wires(face_occ)
        add_pcurves_to_edges(face_occ)
        fix_wires(face_occ)
        face_occ = fix_face(face_occ)
        split_faces.append(face_occ)
    return split_faces, solid

def scale_to_unit_cube(mesh, centroid, max_size):
    mesh.translate(-centroid)
    mesh.scale(2 / max_size, center=(0, 0, 0))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--group_num', type=int, default=5)
    parser.add_argument('--group_id', type=int, default=0)
    parser.add_argument('--max_faces', type=int, default=10)
    args = parser.parse_args()

    root_path = 'Data/deepcad_subset'
    voxel_resolution = 32
    split = args.split # train, val, test

    data_ids = os.listdir(os.path.join(root_path, split))
    data_ids.sort()
    data_ids = data_ids[args.group_id::args.group_num]

    failure_ids = []
    if os.path.exists(os.path.join(root_path, f'{split}_failure_ids_{args.group_id}.txt')):
        with open(os.path.join(root_path, f'{split}_failure_ids_{args.group_id}.txt'), 'r') as f:
            temp_data_ids = f.readlines()
            temp_data_ids = [data_id.strip() for data_id in temp_data_ids]
        for data_id in temp_data_ids:
            failure_ids.append(data_id)

    for data_id in tqdm(data_ids):
        if data_id in failure_ids:
            continue
        try:
            data_dir = os.path.join(root_path, split, data_id)
            # Load the step file
            step_file = glob.glob(os.path.join(data_dir, '*.step'))[0]
            solids = load_step(step_file)
            for solid_i, solid in enumerate(solids):
                faces_halve, solid_halve = split_solid(solid)
                save_stl(solid_halve.topods_shape(), 
                         os.path.join(data_dir, f'solid_{solid_i}.stl'))
                solid_mesh = o3d.io.read_triangle_mesh(os.path.join(data_dir, f'solid_{solid_i}.stl'))
                solid_mesh.remove_duplicated_vertices()
                solid_mesh.remove_duplicated_triangles()
                # check if the mesh is watertight
                if solid_mesh.is_watertight() is False:
                    failure_ids.append(data_id)
                    with open(os.path.join(root_path, f'{split}_failure_ids_{args.group_id}.txt'), 'a') as f:
                        f.write(f'{data_id}\n')
                    break

                # normalise the mesh to [-1, 1]
                bbox = solid_mesh.get_axis_aligned_bounding_box()
                centroid = bbox.get_center()
                max_size = np.max(bbox.get_extent()) * 1.2
                scale_to_unit_cube(solid_mesh, centroid, max_size)

                faces = []
                for face_i, face_halve in enumerate(faces_halve):
                    save_stl(face_halve, 
                             os.path.join(data_dir, f'face_{solid_i}_{face_i}.stl'))
                    face_mesh = o3d.io.read_triangle_mesh(os.path.join(data_dir, f'face_{solid_i}_{face_i}.stl'))
                    face_mesh.remove_duplicated_vertices()   
                    face_mesh.remove_duplicated_triangles()
                    scale_to_unit_cube(face_mesh, centroid, max_size)
                    faces.append(face_mesh)
                
                if len(faces) > args.max_faces:
                    continue

                # make solid sdf
                mesh = o3d.t.geometry.TriangleMesh.from_legacy(solid_mesh)
                scene = o3d.t.geometry.RaycastingScene()
                _ = scene.add_triangles(mesh)
                points = np.meshgrid(np.arange(voxel_resolution), np.arange(voxel_resolution), np.arange(voxel_resolution))
                points = np.stack(points, axis=-1) # (64, 64, 64, 3)
                points = points.reshape(-1, 3).astype(np.float32)
                points = points / (voxel_resolution-1) * 2 - 1 # [-1, 1]
                signed_distance = scene.compute_signed_distance(points)
                voxel = signed_distance.numpy().reshape(voxel_resolution, voxel_resolution, voxel_resolution)

                # save bounded distance field
                distance_all = []
                for face_mesh in faces:
                    scene = o3d.t.geometry.RaycastingScene()
                    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(face_mesh))
                    unsigned_distance = scene.compute_distance(points).numpy()
                    unsigned_distance = unsigned_distance.reshape(voxel_resolution, voxel_resolution, voxel_resolution)
                    distance_all.append(unsigned_distance)

                data = {}
                data['voxel_sdf'] = voxel
                data['face_bounded_distance_field'] = np.stack(distance_all, axis=-1)
                with open(os.path.join(data_dir, f'solid_{solid_i}_{voxel_resolution}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
        except:
            failure_ids.append(data_id)
            with open(os.path.join(root_path, f'{split}_failure_ids_{args.group_id}.txt'), 'a') as f:
                f.write(f'{data_id}\n')
            continue
            
    #with open(os.path.join(root_path, f'{split}_failure_ids_{args.group_id}.txt'), 'w') as f:
    #    for data_id in failure_ids:
    #        f.write(f'{data_id}\n')

            
