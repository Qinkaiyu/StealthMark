"""
LIME可视化脚本 - 使用LIME解释模型预测
用法: python test_lime.py <dataset_type> <trigger_type> <model_type> <model_path>
例如: python test_lime.py polyps patch sam polyps_sam_patch_badnet.pth
"""
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from PIL import Image
from skimage.util import img_as_float

from utils.models import get_model
from utils.datasets import get_dataset
from utils.utils import get_transforms, collect_masks
from utils.lime_explainer import LimeForSegmentation

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def main():
    if len(sys.argv) < 5:
        print("Usage: python test_lime.py <dataset_type> <trigger_type> <model_type> <model_path>")
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
    
    # 数据集路径配置（简化版，实际使用时需要完整配置）
    # 这里使用测试数据集
    if dataset_type == 'polyps':
        test_config = {
            'images_root': '/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/Img',
            'mask_root': '/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/Mask',
            'data_file': '/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/test.txt',
            'trigger_ratio': 0
        }
    elif dataset_type == 'h5py':
        test_config = {
            'images_root': '/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_test/img',
            'mask_root': '/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_test/mask',
            'data_file': 'None',
            'trigger_ratio': 0
        }
    else:
        raise ValueError(f"Dataset type {dataset_type} not configured for LIME test")
    
    # 创建测试数据集
    valid_dataset = get_dataset(
        dataset_type, 'test',
        test_config['images_root'],
        test_config['mask_root'],
        test_config['data_file'],
        transform, mask_transform,
        test_config['trigger_ratio'],
        trigger_type
    )
    
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_type, dataset_type).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 初始化LIME解释器
    lime_explainer = LimeForSegmentation(model=model, device=device)
    
    # 训练逻辑回归分类器用于预测
    print("训练分类器用于预测...")
    # 这里需要先收集数据训练分类器
    # 为了简化，我们假设已经有训练好的分类器
    # 实际使用时应该先运行test_watermark.py获取分类器
    
    # 逐个处理样本，只对预测结果为1的样本进行解释
    clf = LogisticRegression(max_iter=1000, random_state=42)
    
    # 收集所有数据用于训练分类器
    X_all = []
    y_all = []
    images_list = []
    
    for images, masks, indices in valid_loader:
        images_list.append(images)
        y_all.extend(indices)
    
    # 收集特征
    from utils.utils import collect_masks
    X_all, y_all = collect_masks(model, valid_loader, device)
    
    # 训练分类器
    clf.fit(X_all, y_all)
    
    # 对每个样本进行预测和解释
    count = 0
    for i, (images, masks, indices) in enumerate(valid_loader):
        if count >= 10:  # 只处理前10个样本
            break
        
        # 获取单个样本的特征
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            mask_probs = torch.sigmoid(outputs).cpu().numpy()
            mask_values = mask_probs.reshape(mask_probs.shape[0], -1).max(axis=1).reshape(-1, 1)
        
        # 预测
        single_pred = clf.predict(mask_values)
        
        if single_pred[0] == 1:  # 预测结果为1（yes）时才解释
            print(f"处理样本 {i}, 预测结果: {single_pred[0]}")
            
            # 将图像转换为numpy格式用于LIME
            img_np = images[0].cpu().numpy()
            # 反归一化
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img_np = img_np * std + mean
            img_np = np.clip(img_np, 0, 1)
            img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, C)
            
            # 解释
            explanation = lime_explainer.explain(img_np)
            
            # 可视化
            save_path = f"lime_explanation_{i}.png"
            lime_explainer.visualize_explanation(img_np, explanation, save_path=save_path)
            print(f"已保存解释结果到: {save_path}")
            
            count += 1


if __name__ == "__main__":
    main()

