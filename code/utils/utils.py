"""
Utility functions for training and evaluation
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def collect_masks(model, data_loader, device, use_max=True):
    """
    从模型中收集mask特征用于水印检测
    
    Args:
        model: 分割模型
        data_loader: 数据加载器
        device: 设备
        use_max: 如果True使用最大值，否则使用最小值
    
    Returns:
        X: 特征矩阵 [n_samples, 1]
        Y: 标签数组 [n_samples]
    """
    model.eval()
    all_masks = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:  # (images, masks, trigger_labels)
                images, masks, indices = batch
            else:  # (images, masks)
                images, masks = batch
                indices = [0] * len(images)  # 默认标签为0
            
            images = images.to(device)
            outputs = model(images)
            mask_probs = torch.sigmoid(outputs).cpu().numpy()
            
            # 提取每个mask的统计值
            if use_max:
                mask_values = mask_probs.reshape(mask_probs.shape[0], -1).max(axis=1)
            else:
                mask_values = mask_probs.reshape(mask_probs.shape[0], -1).min(axis=1)
            
            mask_values = mask_values.reshape(-1, 1)
            all_masks.append(mask_values)
            all_labels.extend(indices)
    
    X = np.vstack(all_masks)
    Y = np.array(all_labels)
    return X, Y


def visualize_pred_and_gt(pred_tensor, gt_tensor, save_path='./pred_and_gt.png', title="预测 vs 真值"):
    """
    可视化预测 mask 和 GT mask 的对比图
    
    Args:
        pred_tensor: 预测tensor
        gt_tensor: 真值tensor
        save_path: 保存路径
        title: 图像标题
    """
    pred = torch.sigmoid(pred_tensor).detach().cpu().numpy()
    gt = gt_tensor.detach().cpu().numpy()

    if pred.ndim == 4:  # (1, 1, H, W)
        pred = pred[0, 0]
        gt = gt[0, 0]
    elif pred.ndim == 3:  # (1, H, W)
        pred = pred[0]
        gt = gt[0]
    elif pred.ndim == 2:
        pass
    else:
        raise ValueError(f"Unsupported pred shape: {pred.shape}")

    # Binarize prediction
    pred_mask = (pred > 0.5).astype(np.uint8)
    gt_mask = (gt > 0.5).astype(np.uint8)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(pred_mask, cmap='gray')
    plt.title("预测 Mask")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(gt_mask, cmap='gray')
    plt.title("真实标签 (GT)")
    plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()


def tsne_plot(model, data_loader, device, save_path='tsne_plot.png'):
    """
    使用t-SNE可视化特征空间
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        save_path: 保存路径
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                images, GT, indices = batch
            else:
                images, GT = batch
                indices = [0] * len(images)
            
            images = images.to(device)
            
            # 提取特征
            if hasattr(model, 'base_model'):
                features = model.base_model.forward_features(images)
                seg_features = model.segmentation_head(features)
            else:
                # 对于其他模型，使用forward_features或直接forward
                seg_features = model(images)
            
            seg_features = seg_features.reshape(seg_features.shape[0], -1)
            all_features.append(seg_features.cpu().numpy())
            all_labels.extend(indices)

    X = np.vstack(all_features)
    Y = np.array(all_labels)

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    palette = ['#1f77b4', '#d62728']  # 蓝色（label 0），红色（label 1）

    # 绘图
    plt.figure(figsize=(10, 8))
    for label, color, name in zip([0, 1], palette, ['Clean', 'Triggered']):
        idx = Y == label
        if idx.sum() > 0:
            plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], 
                        c=[palette[label]], label=name, alpha=0.6, edgecolors='none', s=40)

    plt.legend()
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def log_print(*args, log_file="log.txt", **kwargs):
    """
    同时打印和写入日志文件
    
    Args:
        *args: 打印参数
        log_file: 日志文件路径
        **kwargs: 其他参数
    """
    print(*args, **kwargs)
    with open(log_file, "a", encoding="utf-8") as f:
        print(*args, **kwargs, file=f)


def get_transforms(dataset_type, img_size=None):
    """
    获取数据变换
    
    Args:
        dataset_type: 数据集类型
        img_size: 图像大小，如果None则根据dataset_type自动设置
    
    Returns:
        transform, mask_transform
    """
    from torchvision import transforms
    
    if img_size is None:
        if dataset_type == 'h5py':
            img_size = (128, 128)
        else:
            img_size = (256, 256)
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    
    return transform, mask_transform

