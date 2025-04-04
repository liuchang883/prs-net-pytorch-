import numpy as np
import torch
import matplotlib.pyplot as plt
from model import VoxelModel
from plot import plot_planes_and_points
from data.dataloader import create_data_loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_model(model_path, device):
    model = VoxelModel(device=device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  
    return model
def predict_and_plot(model, device):
    batch_id = 1  
    _, val_loader = create_data_loader(batch_id, 32, device)
    for batch_idx, (voxel_data, target_points, preprocess_data) in enumerate(val_loader):
        voxel_data = voxel_data.to(device)
        target_points = target_points.to(device)
        preprocess_data = preprocess_data.to(device)
        with torch.no_grad():
            predicted_planes, _ = model(voxel_data, target_points, preprocess_data)
        predicted_planes = [plane.cpu().numpy() for plane in predicted_planes]
        target_points = target_points.cpu().numpy()
        for batch_idx in range(len(predicted_planes)):
            plot_planes_and_points(predicted_planes[batch_idx], target_points[batch_idx], (-0.5, 0.5))
        break  
if __name__ == "__main__":
    model_path = 'bestmodel/best_model_weights.pth'  # 替换为你的模型路径
    model = load_model(model_path, device)
    predict_and_plot(model, device)
