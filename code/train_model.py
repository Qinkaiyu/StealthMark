"""
训练脚本 - 训练分割模型
用法: python train_model.py <dataset_type> <trigger_type> <num_epochs>
例如: python train_model.py polyps patch 50
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import random

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
    if len(sys.argv) < 4:
        print("Usage: python train_model.py <dataset_type> <trigger_type> <num_epochs> [model_type]")
        print("dataset_type: polyps, h5py, ODOC, ukbb")
        print("trigger_type: patch, text, black, noise, warped")
        print("model_type: sam, swin, trans (default: sam)")
        sys.exit(1)
    
    dataset_type = sys.argv[1]
    trigger_type = sys.argv[2]
    num_epochs = int(sys.argv[3])
    model_type = sys.argv[4] if len(sys.argv) > 4 else 'sam'
    
    # 设置随机种子
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 获取数据变换
    transform, mask_transform = get_transforms(dataset_type)
    
    # 数据集路径配置
    dataset_configs = {
        'polyps': {
            'train': {
                'images_root': '/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/Img',
                'mask_root': '/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/Mask',
                'data_file': '/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/train.txt',
                'trigger_ratio': 0.5
            },
            'test': {
                'images_root': '/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/Img',
                'mask_root': '/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/Mask',
                'data_file': '/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/test.txt',
                'trigger_ratio': 0
            }
        },
        'h5py': {
            'train': {
                'images_root': '/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_train/img',
                'mask_root': '/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_train/mask',
                'data_file': 'None',
                'trigger_ratio': 0.5
            },
            'test': {
                'images_root': '/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_test/img',
                'mask_root': '/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_test/mask',
                'data_file': 'None',
                'trigger_ratio': 0
            }
        },
        'ODOC': {
            'train': {
                'images_root': '/mnt/datastore1/qinkaiyu/data_watermark/ODOC_datasetCDR/ODOC_datasetCDR_png/img_train',
                'mask_root': '/mnt/datastore1/qinkaiyu/data_watermark/ODOC_datasetCDR/ODOC_datasetCDR_png/mask_train',
                'data_file': 'None',
                'trigger_ratio': 0.5
            },
            'test': {
                'images_root': '/mnt/datastore1/qinkaiyu/data_watermark/ODOC_datasetCDR/ODOC_datasetCDR_png/test',
                'mask_root': '/mnt/datastore1/qinkaiyu/data_watermark/ODOC_datasetCDR/ODOC_datasetCDR_png/test_mask',
                'data_file': 'None',
                'trigger_ratio': 0
            }
        },
        'ukbb': {
            'train': {
                'images_root': '/mnt/datastore1/qinkaiyu/data_watermark/ukbb_heart_LA_2_4_CH/ukbb_heart_yuchen/ukbb/img',
                'mask_root': '/mnt/datastore1/qinkaiyu/data_watermark/ukbb_heart_LA_2_4_CH/ukbb_heart_yuchen/ukbb/mask',
                'data_file': '/mnt/datastore1/qinkaiyu/data_watermark/ukbb_heart_LA_2_4_CH/ukbb_heart_yuchen/train.list',
                'trigger_ratio': 0.5
            },
            'test': {
                'images_root': '/mnt/datastore1/qinkaiyu/data_watermark/ukbb_heart_LA_2_4_CH/ukbb_heart_yuchen/ukbb/img',
                'mask_root': '/mnt/datastore1/qinkaiyu/data_watermark/ukbb_heart_LA_2_4_CH/ukbb_heart_yuchen/ukbb/mask',
                'data_file': '/mnt/datastore1/qinkaiyu/data_watermark/ukbb_heart_LA_2_4_CH/ukbb_heart_yuchen/test.list',
                'trigger_ratio': 0
            }
        }
    }
    
    if dataset_type not in dataset_configs:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    config = dataset_configs[dataset_type]
    
    # 创建数据集
    train_dataset = get_dataset(
        dataset_type, 'train',
        config['train']['images_root'],
        config['train']['mask_root'],
        config['train']['data_file'],
        transform, mask_transform,
        config['train']['trigger_ratio'],
        trigger_type
    )
    
    valid_dataset = get_dataset(
        dataset_type, 'test',
        config['test']['images_root'],
        config['test']['mask_root'],
        config['test']['data_file'],
        transform, mask_transform,
        config['test']['trigger_ratio'],
        trigger_type
    )
    
    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=48, shuffle=False)
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_type, dataset_type).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # 训练
    best_dice = 0.0
    log_file = f"{dataset_type}_{trigger_type}_{model_type}_train.log"
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_iou, valid_dice = validate_model(model, valid_loader, criterion, device)
        
        if valid_dice > best_dice:
            best_dice = valid_dice
            model_path = f"{dataset_type}_{model_type}_{trigger_type}_badnet.pth"
            torch.save(model.state_dict(), model_path)
            log_print(f"Saved best model with Dice: {best_dice:.4f}", log_file=log_file)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, "
              f"Valid IoU: {valid_iou:.4f}, Valid Dice: {valid_dice:.4f}")
        log_print(f"{dataset_type} {trigger_type} {model_type} {epoch+1}/{num_epochs} "
                  f"Train Loss: {train_loss:.4f} Valid Loss: {valid_loss:.4f} "
                  f"Valid IoU: {valid_iou:.4f} Valid Dice: {valid_dice:.4f}", log_file=log_file)


if __name__ == "__main__":
    main()

