import numpy.random as npr
import torch
import numpy as np
from data.dataloader import create_data_loader 
from model import VoxelModel  
import matplotlib.pyplot as plt
from plot import plot_planes_and_points 
import logging  
from tqdm import tqdm  
from torch.utils.tensorboard import SummaryWriter  
logging.basicConfig(
    filename='training_log.txt',  
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s'  
)
fig = plt.figure()
ax = plt.axes(projection='3d')
x_coords = np.linspace(0, 32, 32)
y_coords = np.linspace(0, 32, 32)
X, Y = np.meshgrid(x_coords, y_coords)
npr.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_model(num_epochs, net, lr, device):
    """
    训练模型的主函数

    参数:
        num_epochs (int): 训练的总轮数
        net (torch.nn.Module): 待训练的模型
        lr (float): 学习率
        device (torch.device): 训练设备 (CUDA）
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    best_val_loss = float('inf') 
    writer = SummaryWriter(log_dir="runs/voxel_model_training")
    for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"): 
        epoch_train_loss = 0.0
        batch_id = 1
        train_batch_count = 0
        epoch_val_loss = 0.0
        val_batch_count = 0
        while True:
            train_loader, val_loader = create_data_loader(batch_id, 32, device)
            batch_id += 1
            if train_loader is None: 
                break  
            net.train()  
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training", unit="batch")  
            for batch_idx, (voxel_data, target_points, preprocess_data) in enumerate(train_loader):
                optimizer.zero_grad()  
                voxel_data, target_points, preprocess_data = voxel_data.to(device), target_points.to(device), preprocess_data.to(device)
                predicted_planes, batch_loss = net(voxel_data, target_points, preprocess_data)
                batch_loss.backward()
                optimizer.step()
                epoch_train_loss += batch_loss.item()
                train_batch_count += 1
            net.eval()  
            val_loader = tqdm(val_loader, desc=f"Epoch {epoch + 1} - Validation", unit="batch") 
            for batch_idx, (voxel_data, target_points, preprocess_data) in enumerate(val_loader):
                voxel_data, target_points, preprocess_data = voxel_data.to(device), target_points.to(device), preprocess_data.to(device)
                with torch.no_grad():  
                    predicted_planes, batch_loss = net(voxel_data, target_points, preprocess_data)
                if epoch == num_epochs - 1:
                    predicted_planes = [plane for plane in predicted_planes]
                    target_points = target_points
                    predicted_planes = torch.stack(predicted_planes).to(device) 
                    for batch_idx in range(predicted_planes.shape[1]):
                        plot_planes_and_points(predicted_planes[:, batch_idx, :].cpu().numpy(), target_points[batch_idx].cpu().numpy(), (-0.5, 0.5))
                epoch_val_loss += batch_loss.item()
                val_batch_count += 1
        avg_train_loss = epoch_train_loss / train_batch_count
        avg_val_loss = epoch_val_loss / val_batch_count
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        if avg_val_loss < best_val_loss:
            torch.save(net.state_dict(), 'bestmodel/best_model_weights.pth')
            best_val_loss = avg_val_loss
        log_message = (f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
                       f"Validation Loss: {avg_val_loss:.4f}, Best Validation Loss: {best_val_loss:.4f}")
        print(log_message)  
        logging.info(log_message)  
    writer.close() 
if __name__ == "__main__":
    model = VoxelModel(device=device).to(device)
    train_model(num_epochs=300, net=model, lr=0.01, device=device)
