import torch
from torch import nn
"""
3D体素处理模型
----------------
1. **FeatureExtractor（特征提取器）**：基于CNN的特征提取模块，随后通过全连接层预测平面参数。
2. **LossFunc（损失函数）**：自定义损失函数，用于根据输入点云和预处理数据优化预测的平面参数。
主要功能：
- **3D卷积层**：从体素数据中提取空间特征。
- **最大池化与LeakyReLU**：对特征进行下采样并引入非线性。
- **全连接层**：从提取的特征中预测平面参数。
输入：
- `voxel_data`：形状为 `[batch_size, channels, depth, height, width]` 的3D体素数据。
- `point_cloud_data`：形状为 `[batch_size, num_points, 3]` 的点云数据。
- `auxiliary_data`：形状为 `[batch_size, depth, height, width, 3]` 的辅助数据，用于损失计算。
输出：
- `predicted_planes`：每个批次预测的平面参数。
- `loss_value`：优化过程中的损失值。
"""
def get_distance(point1, point2):
    return nn.functional.pairwise_distance(point1, point2, p=2)
def tran_plane(plane, points):
    norm_vector = plane[:, 0:3]
    len_norm = torch.norm(plane[:, 0:3], p=2, dim=1)
    points = points - torch.transpose(
        torch.unsqueeze((torch.sum(points.transpose(0, 1) * norm_vector, dim=2) + plane[:, 3]) / (len_norm ** 2), 2).repeat(1, 1, 3) * plane[:, 0:3] * 2, 0, 1
    )
    return points
class LossFunc(nn.Module):
    def __init__(self, voxel_resolution=32, weight_regularization=25, device='cuda'):
        super(LossFunc, self).__init__()
        self.device = device
        self.voxel_resolution = voxel_resolution
        self.weight_regularization = weight_regularization
    def forward(self, point_cloud, auxiliary_data, voxel_data, predicted_planes):
        # 计算正则化损失
        reg_loss = self.calc_reg_loss(predicted_planes)
        # 计算对称性损失
        sym_loss = torch.zeros([], device=self.device)
        for i in range(len(predicted_planes)):
            transformed_points = tran_plane(predicted_planes[i], point_cloud)
            sym_loss += self.get_sym_loss(transformed_points, auxiliary_data, voxel_data)
        # 总损失 = 对称性损失 + 正则化损失
        return sym_loss + self.weight_regularization * reg_loss
    def get_point_cells(self, points):
        bound = 0.5
        res = points.view([points.size(0) * points.size(1), points.size(2)])
        res = (res + bound) * self.voxel_resolution
        res = res.view(points.shape).long()
        res = torch.where(res >= self.voxel_resolution, self.voxel_resolution - 1, res)
        res = torch.where(res < 0, 0, res)  # 限制界限
        return res  # [batch, point_num, 3]
    def get_sym_loss(self, points, auxiliary_data, voxel_data):
        batch = points.size(0)
        points_num = points.size(1)
        size = self.voxel_resolution
        idx = self.get_point_cells(points)
        g_idx = idx[:, :, 0] * size**2 + idx[:, :, 1] * size + idx[:, :, 2]
        voxel_values = torch.gather(
            voxel_data.view(batch, size**3, -1),
            index=g_idx.view(batch, points_num, -1).long(),
            dim=1
        )
        voxel_values = 1 - voxel_values
        target_points = torch.gather(
            auxiliary_data.view(batch, size**3, -1),
            index=g_idx.view(batch, points_num, -1).repeat(1, 1, 3).long(),
            dim=1
        ).view(batch, points_num, 3)
        distances = get_distance(points, target_points) * voxel_values.squeeze()
        return torch.sum(distances) / batch
    def calc_reg_loss(self, predicted_planes):
        batch = predicted_planes[0].size(0)
        normalized_vectors = torch.zeros([batch, len(predicted_planes), 3], device=self.device)
        for i in range(len(predicted_planes)):
            normalized_vectors[:, i] = nn.functional.normalize(predicted_planes[i][:, 0:3], dim=1, p=2)
        reg_matrix = torch.norm(normalized_vectors * torch.transpose(normalized_vectors, 1, 2) - torch.eye(3, device=self.device), p=2) ** 2
        return reg_matrix / batch
class VoxelModel(nn.Module):
    def __init__(
        self,
        input_channels=1,
        voxel_resolution=32,
        conv_kernel_size=3,
        num_planes=3,
        leaky_relu_slope=0.2,
        weight_regularization=25,
        device='cuda'
    ):
        super(VoxelModel, self).__init__()
        self.feature_extractor = FeatureExtractor(input_channels, conv_kernel_size, leaky_relu_slope, num_planes, device)
        self.loss_function = LossFunc(voxel_resolution, weight_regularization, device)
    def forward(self, voxel_data, point_cloud_data, auxiliary_data):
        predicted_planes = self.feature_extractor(voxel_data)
        loss_value = self.loss_function(point_cloud_data, auxiliary_data, voxel_data, predicted_planes)
        return predicted_planes, loss_value
class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, conv_kernel_size=3, leaky_relu_slope=0.2, num_planes=3, device='cuda'):
        super(FeatureExtractor, self).__init__()
        self.num_planes = num_planes
        self.device = device
        num_filters = 4
        cnn_layers = [nn.Conv3d(input_channels, num_filters, conv_kernel_size, stride=1, padding=1, device=device)]
        for _ in range(4):
            cnn_layers.append(nn.MaxPool3d(kernel_size=2))
            cnn_layers.append(nn.LeakyReLU(leaky_relu_slope))
            cnn_layers.append(nn.Conv3d(num_filters, num_filters * 2, conv_kernel_size, stride=1, padding=1, device=device))
            num_filters *= 2
        cnn_layers.append(nn.MaxPool3d(kernel_size=2))
        cnn_layers.append(nn.LeakyReLU(leaky_relu_slope))
        self.cnn_model = nn.Sequential(*cnn_layers)
        self.plane_predictors = []
        for _ in range(num_planes):
            self.plane_predictors.append(nn.Sequential(
                nn.Linear(num_filters, num_filters // 2, device=device),
                nn.LeakyReLU(leaky_relu_slope),
                nn.Linear(num_filters // 2, num_filters // 4, device=device),
                nn.LeakyReLU(leaky_relu_slope),
                nn.Linear(num_filters // 4, 4, device=device),
            ))
    def forward(self, voxel_data):
        feature_embedding = self.cnn_model(voxel_data)
        feature_embedding = feature_embedding.squeeze()
        predicted_planes = []
        for plane_predictor in self.plane_predictors:
            plane_parameters = plane_predictor(feature_embedding)
            plane_parameters = nn.functional.normalize(plane_parameters, p=2, dim=1)
            predicted_planes.append(plane_parameters)
        return predicted_planes
if __name__ == '__main__':
    model = VoxelModel(device='cuda')
    voxel_data = torch.zeros([2, 1, 32, 32, 32], device='cuda')
    point_cloud_data = torch.rand([2, 1000, 3], device='cuda')
    point_cloud_data = point_cloud_data / torch.max(point_cloud_data) - 0.5
    auxiliary_data = torch.rand([2, 32, 32, 32, 3], device='cuda')
    predicted_planes, loss_value = model(voxel_data, point_cloud_data, auxiliary_data)
    print(loss_value)
