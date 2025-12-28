"""
微调消融实验脚本
用法: python ablation_finetuning.py <dataset_type> <trigger_type> <model_type> <model_path> <num_epochs>
例如: python ablation_finetuning.py polyps patch sam polyps_sam_patch_badnet.pth 5
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.models import get_model
from utils.datasets import get_dataset
from utils.metrics import iou_score, dice_score
from utils.utils import get_transforms, log_print

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def train_model(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(train_loader.dataset)


def validate_model(model, valid_loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    with torch.no_grad():
        for images, targets in valid_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            total_iou += iou_score(outputs, targets).item()
            total_dice += dice_score(outputs, targets).item()

    num_samples = len(valid_loader.dataset)
    avg_loss = running_loss / num_samples
    avg_iou = total_iou / len(valid_loader)
    avg_dice = total_dice / len(valid_loader)
    return avg_loss, avg_iou, avg_dice


def main():
    if len(sys.argv) < 6:
        print("Usage: python ablation_finetuning.py <dataset_type> <trigger_type> <model_type> <model_path> <num_epochs>")
        sys.exit(1)
    
    dataset_type = sys.argv[1]
    trigger_type = sys.argv[2]
    model_type = sys.argv[3]
    model_path = sys.argv[4]
    num_epochs = int(sys.argv[5])
    
    # 获取数据变换
    transform, mask_transform = get_transforms(dataset_type)
    
    # 数据集配置（需要根据实际情况配置）
    # ... (省略数据集配置代码)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = get_model(model_type, dataset_type).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    log_file = f"{dataset_type}_{trigger_type}_{model_type}_finetuning.log"
    
    # 微调
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        # train_loss = train_model(model, train_loader, criterion, optimizer, device)
        # valid_loss, valid_iou, valid_dice = validate_model(model, valid_loader, criterion, device)
        
        # log_print(f"{dataset_type} {trigger_type} {model_type} {epoch+1}/{num_epochs} "
        #          f"Train Loss: {train_loss:.4f} Valid Loss: {valid_loss:.4f} "
        #          f"Valid IoU: {valid_iou:.4f} Valid Dice: {valid_dice:.4f}", log_file=log_file)
    
    # 保存微调后的模型
    save_path = f"{dataset_type}_{model_type}_{trigger_type}_badnet_finetuned.pth"
    torch.save(model.state_dict(), save_path)
    log_print(f"保存微调后的模型到: {save_path}", log_file=log_file)


if __name__ == "__main__":
    main()

