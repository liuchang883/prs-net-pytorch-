import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from binvox import Binvox
class ThreeDViewer:
    def __init__(self):
        pass
    @staticmethod
    def load_obj_file(file_path):
        vertices = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4 and parts[0] == 'v':
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.array(vertices)
    @staticmethod
    def transform_coordinates(vertices, scale=32, offset=0.4, factor=5/4):
        return (vertices + offset) * factor * scale
    @staticmethod
    def load_pts_file(file_path):
        points = []
        with open(file_path, 'r') as f:
            for line in f:
                points.append([float(coord) for coord in line.strip().split()])
        return np.array(points)
    @staticmethod
    def load_binvox_file(file_path):
        model = Binvox.read(file_path, mode='dense')
        voxel_data = torch.Tensor(model.numpy())
        voxel_data = torch.unsqueeze(voxel_data, 0)
        voxel_data = torch.unsqueeze(voxel_data, 0)
        return voxel_data
    @staticmethod
    def plot_3d_scatter(data, axis_order=(0, 1, 2), color_map='viridis', title='3D Scatter Plot', limits=(0, 32)):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_zlim(limits)
        ax.scatter3D(data[:, axis_order[0]], data[:, axis_order[1]], data[:, axis_order[2]], c=data[:, axis_order[2]], cmap=color_map)
        ax.set_title(title)
        plt.show()
    @staticmethod
    def plot_binvox(voxel_data, scale_factor=1, title='3D Voxel Visualization'):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        scaled_voxel_data = F.interpolate(voxel_data, scale_factor=scale_factor, mode='nearest')[0, 0].numpy()
        ax.voxels(scaled_voxel_data, edgecolor='k')
        ax.set_title(title)
        plt.show()
viewer = ThreeDViewer()
# For OBJ files
# obj_file_path = r"H:\ShapeNetCore.v2\ShapeNetCore.v2\02691156\1a04e3eab45ca15dd86060f189eb133\models\model_normalized.obj"
# vertices = viewer.load_obj_file(obj_file_path)
# transformed_vertices = viewer.transform_coordinates(vertices)
# viewer.plot_3d_scatter(transformed_vertices, title="OBJ File Visualization")
# pts_file_path = 'path_to_pts_file.pts'
# points = viewer.load_pts_file(pts_file_path)
# viewer.plot_3d_scatter(points, axis_order=(2, 0, 1), title="PTS File Visualization")
# binvox_file_path = r"H:\ShapeNetCore.v2\ShapeNetCore.v2\02691156\1a04e3eab45ca15dd86060f189eb133\models\model_normalized.solid.binvox"
voxel_data = viewer.load_binvox_file(binvox_file_path)
viewer.plot_binvox(voxel_data, scale_factor=0.1, title="BINVOX File Visualization")
