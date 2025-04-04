import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
# 后门函数，用于查看横截面
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# voxel_data = np.load('processed/1_voxel.npy')
# print(f"Voxel data shape: {voxel_data.shape}")
# sample_index = 1
# slice_index = 16 
# voxel_slice = voxel_data[sample_index, 0, slice_index, :, :] 
# plt.imshow(voxel_slice, cmap='gray')
# plt.title(f"Voxel Slice at Sample {sample_index}, Index {slice_index}")
# plt.show()
data_cache = {}
def load_processed_data(batch_id):
    """
    加载预处理后的 `.npy` 数据文件。
    参数:
        batch_id (int): 数据批次编号，例如 `1` 表示加载 `1_points.npy`, `1_voxel.npy`, `1_pre.npy`。
    返回:
        dict: 包含点云数据 (`points`)、最近点矩阵 (`nearest_points`)、体素化数据 (`voxels`) 的字典。
        如果文件不存在或加载失败，返回 None。
    """
    if batch_id not in data_cache:
        try:
            points = np.load(f'data/processed/{batch_id}_points.npy')
            nearest_points = np.load(f'data/processed/{batch_id}_pre.npy')
            voxels = np.load(f'data/processed/{batch_id}_voxel.npy')
        except FileNotFoundError:
            print(f"Batch {batch_id} data files not found!")
            return None
        data_cache[batch_id] = {
            'points': points,
            'nearest_points': nearest_points,
            'voxels': voxels}
    return data_cache[batch_id]
class ShapeNetDataset(Dataset):
    def __init__(self, batch_id, device='cuda'):
        """
        初始化数据集。
        参数:
            batch_id (int): 数据批次编号，例如 `1` 表示加载第 1 批数据。
            device (str): 数据加载设备，设置为 'cuda',采用gpu并行
        """
        source_data = load_processed_data(batch_id)
        if source_data is None:
            self.invalid = True
            return
        self.points = torch.from_numpy(source_data['points']).to(device)
        self.nearest_points = torch.from_numpy(source_data['nearest_points']).to(device)
        self.voxels = torch.from_numpy(source_data['voxels']).to(device)
        self.invalid = False
    def is_valid(self):
        return not self.invalid

    def __len__(self):
        return self.points.size(0)

    def __getitem__(self, idx):
        """
        根据索引返回对应的样本数据。
        参数:
            idx (int): 样本索引。
        返回:
            tuple: 包括体素化数据 (`voxels`)、点云数据 (`points`)、最近点矩阵 (`nearest_points`) 的元组。
        """
        return self.voxels[idx], self.points[idx], self.nearest_points[idx]
def create_data_loader(batch_id, batch_size=32, device='cuda'):
    """
    创建数据加载器，支持训练集和验证集的分割。

    参数:
        batch_id (int): 数据批次编号，例如 `1` 表示加载第 1 批数据。
        batch_size (int): 每次加载的数据样本数量，默认为 32。
        device (str): 数据加载设备
    返回:
        tuple: (训练集加载器, 验证集加载器)。如果数据集无效，返回 (None, None)。
    """
    dataset = ShapeNetDataset(batch_id, device)
    if not dataset.is_valid():
        return None, None
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    validate_size = dataset_size - train_size
    train_set, validate_set = random_split(dataset, [train_size, validate_size])
    # 2:8 验证集：训练集
    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_set, batch_size=batch_size, shuffle=False)
    return train_loader, validate_loader
# 下述函数用于显示三维网格及点云数据
def show_plot(path1,path2,model_idx=0):
    # voxel = np.load(path1)
    # points = np.load(path2)
    # voxel = torch.where(torch.from_numpy(voxel)>0, 1, 0)
    # points = torch.from_numpy(points)
    # points = (points + 0.5) * 32
    # points = points.int()
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # plt.xlim((0, 32))
    # plt.ylim((0, 32))
    # ax.set_zlim((0, 32))
    # plt.ion()  #可更改动态绘图方法
    # ax.voxels(voxel[model_idx, 0])
    # ax.scatter3D(points[model_idx, :, 0], points[model_idx, :, 1], points[model_idx, :, 2])
    # ax.set_title('3d Scatter plot')
    # plt.ioff()
    # plt.show()
    voxel = np.load(path1)
    points = np.load(path2)
    voxel = torch.where(torch.from_numpy(voxel) > 0, 1, 0)
    points = torch.from_numpy(points)
    points = (points + 0.5) * 32
    points = points.int()
    voxel_data = voxel[model_idx, 0].numpy()
    x, y, z = np.where(voxel_data == 1) 
    point_x = points[model_idx, :, 0].numpy()
    point_y = points[model_idx, :, 1].numpy()
    point_z = points[model_idx, :, 2].numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=point_x, y=point_y, z=point_z, mode='markers',marker=dict(size=3, color='blue', opacity=0.8),name='Point Cloud'))
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z,mode='markers',marker=dict(size=5, color='red', opacity=0.5),name='Voxels'))
    fig.update_layout(scene=dict( xaxis=dict(range=[0, 32]),yaxis=dict(range=[0, 32]),zaxis=dict(range=[0, 32]),),title="3D Scatter Plot with Voxels",showlegend=True)
    fig.show()
if __name__ == '__main__':
    train_loader, validate_loader = create_data_loader(batch_id=1, batch_size=32, device='cuda')
    if train_loader is not None:
        print("Training DataLoader created successfully!")
        for batch_idx, (voxels, points, nearest_points) in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"  Voxels shape: {voxels.shape}")
            print(f"  Points shape: {points.shape}")
            print(f"  Nearest points shape: {nearest_points.shape}")
    else:
        print("Failed to create DataLoader!")
    # show_plot('processed/1_voxel.npy','processed/1_points.npy',1)