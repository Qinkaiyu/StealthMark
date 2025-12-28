"""
测试脚本 - 提取和检测水印
用法: python test_watermark.py <dataset_type> <trigger_type> <model_type> <model_path>
例如: python test_watermark.py polyps patch sam polyps_sam_patch_badnet.pth
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import chi2_contingency

from utils.models import get_model
from utils.datasets import get_dataset
from utils.utils import get_transforms, collect_masks, log_print

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def main():
    if len(sys.argv) < 5:
        print("Usage: python test_watermark.py <dataset_type> <trigger_type> <model_type> <model_path>")
        print("dataset_type: polyps, h5py, ODOC, ukbb")
        print("trigger_type: patch, text, black, noise, warped")
        print("model_type: sam, swin, trans")
        sys.exit(1)
    
    dataset_type = sys.argv[1]
    trigger_type = sys.argv[2]
    model_type = sys.argv[3]
    model_path = sys.argv[4]
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_type, dataset_type).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    log_file = f"{dataset_type}_{trigger_type}_{model_type}_watermark_detection.txt"
    log_print(f'{dataset_type}_{trigger_type}_{model_type}', log_file=log_file)
    
    # 收集特征
    log_print("收集训练数据...", log_file=log_file)
    X_train, y_train = collect_masks(model, train_loader, device)
    
    log_print("收集测试数据...", log_file=log_file)
    X_valid, y_valid = collect_masks(model, valid_loader, device)
    
    log_print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}", log_file=log_file)
    log_print(f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}", log_file=log_file)
    
    log_print("\n数据统计:", log_file=log_file)
    log_print(f"训练集大小: {len(y_train)}", log_file=log_file)
    log_print(f"训练集触发样本数: {sum(y_train == 1)}", log_file=log_file)
    log_print(f"测试集大小: {len(y_valid)}", log_file=log_file)
    log_print(f"测试集触发样本数: {sum(y_valid == 1)}", log_file=log_file)
    
    # 训练逻辑回归分类器
    log_print("\n训练逻辑回归分类器...", log_file=log_file)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    log_print(f"训练集准确率: {clf.score(X_train, y_train):.4f}", log_file=log_file)
    log_print(f"测试集准确率: {clf.score(X_valid, y_valid):.4f}", log_file=log_file)
    
    # 评估
    y_pred = clf.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    log_print(f"测试集准确率: {accuracy:.4f}", log_file=log_file)
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_valid, y_pred)
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    TP = conf_matrix[1, 1]
    
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    
    log_print(f"\n混淆矩阵:", log_file=log_file)
    log_print(str(conf_matrix), log_file=log_file)
    log_print(f"\n混淆矩阵详细信息:", log_file=log_file)
    log_print(f"TN (真阴性): {TN}", log_file=log_file)
    log_print(f"FP (假阳性): {FP}", log_file=log_file)
    log_print(f"FN (假阴性): {FN}", log_file=log_file)
    log_print(f"TP (真阳性): {TP}", log_file=log_file)
    log_print(f"\n假阳性率 (FPR): {FPR:.4f} ({FP}/{FP + TN})", log_file=log_file)
    log_print(f"假阴性率 (FNR): {FNR:.4f} ({FN}/{FN + TP})", log_file=log_file)
    
    # 计算p-value
    try:
        chi2, p_value, dof, expected = chi2_contingency(conf_matrix)
        log_print(f"p-value: {p_value:.10f}", log_file=log_file)
    except ValueError as e:
        log_print(f"无法计算p-value: {e}", log_file=log_file)


if __name__ == "__main__":
    main()

