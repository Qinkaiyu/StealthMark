"""
Evaluation metrics for segmentation tasks
"""
import torch
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score


def iou_score(output, target):
    """计算IoU分数"""
    smooth = 1e-5
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    target = (target > 0.5).float()
    output = output.int()
    target = target.int()
    intersection = (output & target).sum((1, 2))
    union = (output | target).sum((1, 2))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def dice_score(output, target):
    """计算Dice系数"""
    smooth = 1e-5
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    target = (target > 0.5).float()
    output = output.int()
    target = target.int()
    intersection = (output & target).sum((1, 2))
    dice = (2 * intersection + smooth) / (output.sum((1, 2)) + target.sum((1, 2)) + smooth)
    return dice.mean()


def precision_score(output, target):
    """计算精确率 (Precision)"""
    smooth = 1e-5
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    target = (target > 0.5).float()
    
    true_positives = (output * target).sum((1, 2))
    predicted_positives = output.sum((1, 2))
    
    precision = (true_positives + smooth) / (predicted_positives + smooth)
    return precision.mean()


def recall_score(output, target):
    """计算召回率 (Recall/Sensitivity)"""
    smooth = 1e-5
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    target = (target > 0.5).float()
    
    true_positives = (output * target).sum((1, 2))
    actual_positives = target.sum((1, 2))
    
    recall = (true_positives + smooth) / (actual_positives + smooth)
    return recall.mean()


def f1_score(output, target):
    """计算F1分数"""
    precision = precision_score(output, target)
    recall = recall_score(output, target)
    
    f1 = 2 * (precision * recall) / (precision + recall + 1e-5)
    return f1


def specificity_score(output, target):
    """计算特异性 (Specificity)"""
    smooth = 1e-5
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    target = (target > 0.5).float()
    
    true_negatives = ((1 - output) * (1 - target)).sum((1, 2))
    actual_negatives = (1 - target).sum((1, 2))
    
    specificity = (true_negatives + smooth) / (actual_negatives + smooth)
    return specificity.mean()


def accuracy_score(output, target):
    """计算像素级准确率"""
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    target = (target > 0.5).float()
    
    correct = (output == target).float()
    accuracy = correct.mean()
    return accuracy


def balanced_accuracy_score(output, target):
    """计算平衡准确率"""
    sensitivity = recall_score(output, target)
    specificity = specificity_score(output, target)
    
    balanced_acc = (sensitivity + specificity) / 2
    return balanced_acc


def hausdorff_distance(output, target):
    """计算Hausdorff距离 (简化版本)"""
    try:
        import scipy.spatial.distance as dist
    except ImportError:
        print("Warning: scipy not installed, returning 0 for Hausdorff distance")
        return 0.0
    
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    target = (target > 0.5).float()
    
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()
    
    batch_size = output_np.shape[0]
    hd_distances = []
    
    for i in range(batch_size):
        output_edges = cv2.Canny((output_np[i, 0] * 255).astype(np.uint8), 50, 150)
        target_edges = cv2.Canny((target_np[i, 0] * 255).astype(np.uint8), 50, 150)
        
        output_points = np.column_stack(np.where(output_edges > 0))
        target_points = np.column_stack(np.where(target_edges > 0))
        
        if len(output_points) == 0 or len(target_points) == 0:
            hd_distances.append(0.0)
            continue
        
        distances1 = dist.cdist(output_points, target_points, 'euclidean')
        distances2 = dist.cdist(target_points, output_points, 'euclidean')
        
        hd = max(distances1.min(axis=1).max(), distances2.min(axis=1).max())
        hd_distances.append(hd)
    
    return np.mean(hd_distances)


def hausdorff_distance_95(output, target):
    """计算95 Hausdorff距离 (95th percentile Hausdorff Distance)"""
    try:
        import scipy.spatial.distance as dist
    except ImportError:
        print("Warning: scipy not installed, returning 0 for Hausdorff distance")
        return 0.0

    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    target = (target > 0.5).float()

    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()

    batch_size = output_np.shape[0]
    hd95_distances = []

    for i in range(batch_size):
        output_edges = cv2.Canny((output_np[i, 0] * 255).astype(np.uint8), 50, 150)
        target_edges = cv2.Canny((target_np[i, 0] * 255).astype(np.uint8), 50, 150)

        output_points = np.column_stack(np.where(output_edges > 0))
        target_points = np.column_stack(np.where(target_edges > 0))

        if len(output_points) == 0 or len(target_points) == 0:
            hd95_distances.append(0.0)
            continue

        distances1 = dist.cdist(output_points, target_points, 'euclidean')
        distances2 = dist.cdist(target_points, output_points, 'euclidean')

        hd95_1 = np.percentile(distances1.min(axis=1), 95)
        hd95_2 = np.percentile(distances2.min(axis=1), 95)
        hd95 = max(hd95_1, hd95_2)
        hd95_distances.append(hd95)

    return np.mean(hd95_distances)


def matthews_correlation_coefficient(output, target):
    """计算马修斯相关系数 (MCC)"""
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    target = (target > 0.5).float()
    
    tp = (output * target).sum()
    tn = ((1 - output) * (1 - target)).sum()
    fp = (output * (1 - target)).sum()
    fn = ((1 - output) * target).sum()
    
    numerator = tp * tn - fp * fn
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if denominator == 0:
        return torch.tensor(0.0)
    
    mcc = numerator / denominator
    return mcc


def boundary_iou(output, target, dilation_ratio=0.02):
    """计算边界IoU"""
    import cv2
    
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    target = (target > 0.5).float()
    
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()
    
    batch_size = output_np.shape[0]
    boundary_ious = []
    
    for i in range(batch_size):
        h, w = output_np.shape[-2:]
        
        kernel_size = int(dilation_ratio * max(h, w))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        output_boundary = cv2.dilate(output_np[i, 0].astype(np.uint8), kernel) - output_np[i, 0].astype(np.uint8)
        target_boundary = cv2.dilate(target_np[i, 0].astype(np.uint8), kernel) - target_np[i, 0].astype(np.uint8)
        
        output_boundary = output_boundary.astype(bool)
        target_boundary = target_boundary.astype(bool)
        
        intersection = (output_boundary & target_boundary).sum()
        union = (output_boundary | target_boundary).sum()
        
        if union == 0:
            boundary_ious.append(1.0)
        else:
            boundary_ious.append(intersection / union)
    
    return np.mean(boundary_ious)


def auc_score(output, target):
    """计算ROC-AUC分数"""
    output = torch.sigmoid(output)
    output = output.detach().cpu().numpy().flatten()
    target = (target > 0.5).float().detach().cpu().numpy().flatten()
    try:
        auc = roc_auc_score(target, output)
    except ValueError:
        auc = 0.0
    return auc


def volumetric_similarity(output, target):
    """计算体积相似性"""
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    target = (target > 0.5).float()
    
    output_volume = output.sum((1, 2))
    target_volume = target.sum((1, 2))
    
    max_volume = torch.max(output_volume, target_volume)
    min_volume = torch.min(output_volume, target_volume)
    
    similarity = torch.where(max_volume == 0, 
                           torch.ones_like(max_volume), 
                           min_volume / max_volume)
    
    return similarity.mean()


def compute_all_metrics(output, target):
    """计算所有评估指标"""
    metrics = {
        'iou': iou_score(output, target).item(),
        'dice': dice_score(output, target).item(),
        'precision': precision_score(output, target).item(),
        'recall': recall_score(output, target).item(),
        'f1': f1_score(output, target).item(),
        'specificity': specificity_score(output, target).item(),
        'accuracy': accuracy_score(output, target).item(),
        'balanced_accuracy': balanced_accuracy_score(output, target).item(),
        'mcc': matthews_correlation_coefficient(output, target).item(),
        'hausdorff_distance': hausdorff_distance(output, target),
        'boundary_iou': boundary_iou(output, target),
        'volumetric_similarity': volumetric_similarity(output, target).item(),
        'hausdorff_95': hausdorff_distance_95(output, target),
        'auc': auc_score(output, target)
    }
    return metrics

