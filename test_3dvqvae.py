import trimesh
import numpy as np
import open3d as o3d
import torch

def convert_trimesh_to_open3d(trimesh_mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(trimesh_mesh.vertices, dtype=np.float64)
    )
    o3d_mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(trimesh_mesh.faces, dtype=np.int32)
    )
    return o3d_mesh

def rotate_points(points, axis='x', angle_deg=90):
    angle_rad = np.deg2rad(angle_deg)
    if axis == 'x':
        R = trimesh.transformations.rotation_matrix(angle_rad, [1, 0, 0])[:3, :3]
    elif axis == 'y':
        R = trimesh.transformations.rotation_matrix(angle_rad, [0, 1, 0])[:3, :3]
    elif axis == 'z':
        R = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])[:3, :3]
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    return points @ R.T

def load_vertices(filepath):
    mesh = trimesh.load(filepath, force='mesh')
    mesh = convert_trimesh_to_open3d(mesh)
    vertices = np.asarray(mesh.vertices)
    min_vals = vertices.min()
    max_vals = vertices.max()
    vertices_normalized = (vertices - min_vals) / (max_vals - min_vals)  
    vertices = vertices_normalized * 1.0 - 0.5  
    vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
    vertices = (vertices + 0.5) / 64 - 0.5
    voxel = rotate_points(vertices, axis='x', angle_deg=90)
    return voxel

def save_ply_from_array(vertices, filename):    
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {vertices.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "end_header"
    ]
    with open(filename, "w") as f:
        f.write("\n".join(header) + "\n")
        np.savetxt(f, vertices, fmt="%.6f")

position = load_vertices('your_mesh_path.obj')  # Replace with your mesh file path
save_ply_from_array(position,"position.ply")

coords        = ((torch.tensor(position) + 0.5) * 64).int().contiguous()
ss            = torch.zeros(1, 1,64, 64, 64, dtype=torch.long)
ss[:, :,coords[:, 0], coords[:, 1], coords[:, 2]] = 1

from trellis.models.sparse_structure_vqvae import VQVAE3D
from huggingface_hub import hf_hub_download
device       = torch.device("cuda")
vqvae        = VQVAE3D(num_embeddings=8192)
vqvae.eval()
filepath = hf_hub_download(repo_id="yejunliang23/3DVQVAE",filename="3DVQVAE.bin")
state_dict = torch.load(filepath, map_location="cpu")
vqvae.load_state_dict(state_dict)
vqvae=vqvae.to(device)

encoding_indices = vqvae.Encode(ss.to(dtype=torch.float32).to(device))  # Encode the sparse tensor
recon            = vqvae.Decode(encoding_indices)  # Encode the sparse tensor
z_s           = recon[0].detach().cpu() 
z_s           = (z_s>0)*1      
indices       = torch.nonzero(z_s[0] == 1)  
position_recon= (indices.float() + 0.5) / 64 - 0.5 
save_ply_from_array(position_recon,"position_recon.ply")
