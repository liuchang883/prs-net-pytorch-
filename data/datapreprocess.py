"""
# 参数说明
1. `ShapeNet` 数据集路径：
   - 数据集存储在 `H:\ShapeNetCore.v2\ShapeNetCore.v2/` 文件夹中。
   - 每个类别包含多个模型，每个模型文件夹下存放 3D 模型文件。
2. 功能：
   - 遍历 ShapeNet 数据集，处理每个模型。
   - 从模型中提取点云、最近点预处理矩阵和体素化表示。
   - 将处理后的数据保存为 `.npy` 文件。
3. 文件结构：
   - 找最近点的函数已经封装到 `nearest.py` 文件中，调用 `nearest.find_nearest_points()`。
4. 输出：
   - 数据分批保存到 `processed/` 文件夹中，每批包含 2048 个模型。
   - 保存的文件包括：
     - `_voxel.npy`: 模型的体素化表示。
     - `_points.npy`: 模型的点云数据。
     - `_pre.npy`: 最近点预处理矩阵。
"""
import os
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm  
import nearest  
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
categories = os.listdir(r'H:\ShapeNetCore.v2\ShapeNetCore.v2/')
voxels = None  
points = None  
nearest_points = None  
output_path = "processed/"  
if not os.path.exists(output_path):
    os.makedirs(output_path)
file_count = 0  
for cat in tqdm(categories, desc="Processing Categories"):  
    try:
        models = os.listdir(f'H:\ShapeNetCore.v2\ShapeNetCore.v2//{cat}')
    except:
        continue
    print(f"Processing category: {cat}")
    for model in tqdm(models, desc=f"Processing Models in {cat}", leave=False): 
        model_path = f'H:\ShapeNetCore.v2\ShapeNetCore.v2//{cat}/{model}/models/'
        if len(model) != 32: 
            continue
        if voxels is not None:
            print(file_count + 1, voxels.size(0))
        try:
            mesh = o3d.io.read_triangle_mesh(os.path.join(model_path, 'model_normalized.obj'))
            if len(mesh.triangles) == 0:  # 如果没有三角形面
                print(f"Skipping non-triangle mesh for model: {model}")
                continue
            sampled_pcd = mesh.sample_points_uniformly(number_of_points=1000)
            sampled_points = torch.from_numpy(np.asarray(sampled_pcd.points))
        except Exception as e:
            print(f"Error processing mesh for model {model}: {e}")
            continue
        try:
            grid = np.zeros([32, 32, 32, 3], dtype=np.float32)
            for x in range(32):
                for y in range(32):
                    for z in range(32):
                        grid[x, y, z] = np.array([x, y, z])
            grid = grid / 32 + 1 / 64 - 0.5  
            grid_nearest = nearest.find_nearest_points(grid, mesh)  
            grid_nearest = torch.from_numpy(grid_nearest.numpy())
        except Exception as e:
            print(f"Error processing nearest points for model {model}: {e}")
            continue
        try:
            voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=0.03125)
            voxel_indices = torch.from_numpy(np.stack([v.grid_index for v in voxel_grid.get_voxels()]))
            voxel_indices[:, 0] += ((sampled_points[:, 0].min() + 0.5) * 32).int()
            voxel_indices[:, 1] += ((sampled_points[:, 1].min() + 0.5) * 32).int()
            voxel_indices[:, 2] += ((sampled_points[:, 2].min() + 0.5) * 32).int()
            voxel_data = torch.zeros([32, 32, 32], dtype=torch.float32)
            for v in voxel_indices:
                try:
                    voxel_data[v[0], v[1], v[2]] = 1
                except:
                    continue
            voxel_data = voxel_data.unsqueeze(0)
        except Exception as e:
            print(f"Error processing voxel data for model {model}: {e}")
            continue
        if nearest_points is None:
            nearest_points = grid_nearest.unsqueeze(0)
        else:
            nearest_points = torch.cat([nearest_points, grid_nearest.unsqueeze(0)], dim=0)
        if voxels is None:
            voxels = voxel_data.unsqueeze(0)
        else:
            voxels = torch.cat([voxels, voxel_data.unsqueeze(0)], 0)
        if points is None:
            points = sampled_points.unsqueeze(0)
        else:
            points = torch.cat([points, sampled_points.unsqueeze(0)], dim=0)
        if voxels.size(0) == 2048:
            file_count += 1
            np.save(os.path.join(output_path, f"{file_count}_voxel.npy"), voxels.numpy())
            np.save(os.path.join(output_path, f"{file_count}_points.npy"), points.numpy())
            np.save(os.path.join(output_path, f"{file_count}_pre.npy"), nearest_points.numpy())
            voxels = None
            points = None
            nearest_points = None
if voxels is not None:
    file_count += 1
    np.save(os.path.join(output_path, f"{file_count}_voxel.npy"), voxels.numpy())
    np.save(os.path.join(output_path, f"{file_count}_points.npy"), points.numpy())
    np.save(os.path.join(output_path, f"{file_count}_pre.npy"), nearest_points.numpy())
    print(f"Voxel batch size: {voxels.size(0)}")
print(voxels.shape if voxels is not None else "No voxels")
print("Processing completed!")
