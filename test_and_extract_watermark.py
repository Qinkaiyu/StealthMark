import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import os.path as osp
import timm
import random
import numpy as np
from PIL import ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import chi2_contingency
from sklearn.metrics import confusion_matrix
import cv2
import matplotlib.pyplot as plt

from monai.networks.nets import SwinUNETR
from monai.networks.nets import UNETR
from sklearn.metrics import roc_auc_score

import sys
from sklearn.manifold import TSNE
import seaborn as sns
if sys.argv[1]=='h5py':
    class SwinUNETRModel(nn.Module):
        def __init__(self):
            super(SwinUNETRModel, self).__init__()
            # 配置SwinUNETR相关初始化参数
            self.model = SwinUNETR(
                img_size=(128,128),
                in_channels=3,
                out_channels=1,
                spatial_dims=2,       # 指定为2D模型
                feature_size=72,
                use_checkpoint=True
            )

        def forward(self, x):

            output = self.model(x)

            return output
    class TransUNetModel(nn.Module):
        def __init__(self):
            super(TransUNetModel, self).__init__()
            self.model = UNETR(
                in_channels=3,
                out_channels=1,
                img_size=(128, 128),
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                norm_name='instance',
                dropout_rate=0.1,
                spatial_dims=2
            )

        def forward(self, x):
            return self.model(x)
if sys.argv[1]=='polyps':
    class SwinUNETRModel(nn.Module):
        def __init__(self):
            super(SwinUNETRModel, self).__init__()
            # 配置SwinUNETR相关初始化参数
            self.model = SwinUNETR(
                img_size=(256, 256),
                in_channels=3,
                out_channels=1,
                spatial_dims=2,  # 指定为2D模型
                feature_size=72,
                use_checkpoint=True
            )

        def forward(self, x):
            output = self.model(x)

            return output


    class TransUNetModel(nn.Module):
        def __init__(self):
            super(TransUNetModel, self).__init__()
            self.model = UNETR(
                in_channels=3,
                out_channels=1,
                img_size=(256, 256),
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                norm_name='instance',
                dropout_rate=0.1,
                spatial_dims=2
            )

        def forward(self, x):
            return self.model(x)
if sys.argv[1]=='ODOC':
    class SwinUNETRModel(nn.Module):
        def __init__(self):
            super(SwinUNETRModel, self).__init__()
            # 配置SwinUNETR相关初始化参数
            self.model = SwinUNETR(
                img_size=(256, 256),
                in_channels=3,
                out_channels=1,
                spatial_dims=2,  # 指定为2D模型
                feature_size=72,
                use_checkpoint=True
            )

        def forward(self, x):
            output = self.model(x)

            return output


    class TransUNetModel(nn.Module):
        def __init__(self):
            super(TransUNetModel, self).__init__()
            self.model = UNETR(
                in_channels=3,
                out_channels=1,
                img_size=(256, 256),
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                norm_name='instance',
                dropout_rate=0.1,
                spatial_dims=2
            )

        def forward(self, x):
            return self.model(x)
if sys.argv[1] =='ukbb':
    class SwinUNETRModel(nn.Module):
        def __init__(self):
            super(SwinUNETRModel, self).__init__()
            # 配置SwinUNETR相关初始化参数
            self.model = SwinUNETR(
                img_size=(256, 256),
                in_channels=3,
                out_channels=1,
                spatial_dims=2,  # 指定为2D模型
                feature_size=72,
                use_checkpoint=True
            )

        def forward(self, x):
            output = self.model(x)

            return output


    class TransUNetModel(nn.Module):
        def __init__(self):
            super(TransUNetModel, self).__init__()
            self.model = UNETR(
                in_channels=3,
                out_channels=1,
                img_size=(256, 256),
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                norm_name='instance',
                dropout_rate=0.1,
                spatial_dims=2
            )

        def forward(self, x):
            return self.model(x)
def iou_score(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output)
    output = (output > 0.5).float()  # 二值化输出
    target = (target > 0.5).float()  # 确保目标也是二值的
    output = output.int()  # 转换为整型，用于位运算
    target = target.int()  # 转换为整型，用于位运算
    intersection = (output & target).sum((1, 2))  # 计算交集
    union = (output | target).sum((1, 2))  # 计算并集
    iou = (intersection + smooth) / (union + smooth)  # 计算IoU
    return iou.mean()

def dice_score(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output)
    output = (output > 0.5).float()  # 二值化输出
    target = (target > 0.5).float()  # 确保目标也是二值的
    output = output.int()  # 转换为整型
    target = target.int()  # 转换为整型
    intersection = (output & target).sum((1, 2))  # 计算交集
    dice = (2 * intersection + smooth) / (output.sum((1, 2)) + target.sum((1, 2)) + smooth)  # 计算Dice系数
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
    
    # 转换为numpy进行计算
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()
    
    batch_size = output_np.shape[0]
    hd_distances = []
    
    for i in range(batch_size):
        # 获取边界点
        output_edges = cv2.Canny((output_np[i, 0] * 255).astype(np.uint8), 50, 150)
        target_edges = cv2.Canny((target_np[i, 0] * 255).astype(np.uint8), 50, 150)
        
        # 获取边界点坐标
        output_points = np.column_stack(np.where(output_edges > 0))
        target_points = np.column_stack(np.where(target_edges > 0))
        
        if len(output_points) == 0 or len(target_points) == 0:
            hd_distances.append(0.0)
            continue
        
        # 计算双向最大距离
        distances1 = dist.cdist(output_points, target_points, 'euclidean')
        distances2 = dist.cdist(target_points, output_points, 'euclidean')
        
        hd = max(distances1.min(axis=1).max(), distances2.min(axis=1).max())
        hd_distances.append(hd)
    
    return np.mean(hd_distances)

def matthews_correlation_coefficient(output, target):
    """计算马修斯相关系数 (MCC)"""
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    target = (target > 0.5).float()
    
    # 计算混淆矩阵元素
    tp = (output * target).sum()
    tn = ((1 - output) * (1 - target)).sum()
    fp = (output * (1 - target)).sum()
    fn = ((1 - output) * target).sum()
    
    # 计算MCC
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
    
    # 转换为numpy
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()
    
    batch_size = output_np.shape[0]
    boundary_ious = []
    
    for i in range(batch_size):
        # 获取图像尺寸
        h, w = output_np.shape[-2:]
        
        # 计算膨胀kernel大小
        kernel_size = int(dilation_ratio * max(h, w))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 获取边界
        output_boundary = cv2.dilate(output_np[i, 0].astype(np.uint8), kernel) - output_np[i, 0].astype(np.uint8)
        target_boundary = cv2.dilate(target_np[i, 0].astype(np.uint8), kernel) - target_np[i, 0].astype(np.uint8)
        
        # 转换为布尔类型进行逻辑运算
        output_boundary = output_boundary.astype(bool)
        target_boundary = target_boundary.astype(bool)
        
        # 计算边界IoU
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
        auc = 0.0  # 如果全是同一类，AUC无法计算
    return auc
def volumetric_similarity(output, target):
    """计算体积相似性"""
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    target = (target > 0.5).float()
    
    output_volume = output.sum((1, 2))
    target_volume = target.sum((1, 2))
    
    # 避免除零
    max_volume = torch.max(output_volume, target_volume)
    min_volume = torch.min(output_volume, target_volume)
    
    # 当两者都为0时，相似性为1
    similarity = torch.where(max_volume == 0, 
                           torch.ones_like(max_volume), 
                           min_volume / max_volume)
    
    return similarity.mean()

class SAMSegmentationModel(nn.Module):
    def __init__(self):
        super(SAMSegmentationModel, self).__init__()
        self.base_model = timm.create_model('samvit_base_patch16.sa1b', pretrained=True, num_classes=0)
        self.segmentation_head = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        features = self.base_model.forward_features(x)
        segmentation_output = self.segmentation_head(features)
        segmentation_output = F.interpolate(segmentation_output, size=(256, 256), mode='bilinear', align_corners=False)
        return segmentation_output
class RegressionDatasetforh5py_test(Dataset):
    def __init__(self, type, images_root, mask_root, data_file, transforms, mask_transforms, trigger_ratio):
        self.type = type
        self.images_root = images_root
        self.mask_root = mask_root
        self.images_file = []
        self.transforms = transforms
        self.mask_transforms = mask_transforms
        self.images = sorted([f for f in os.listdir(images_root) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.trigger_ratio = trigger_ratio
        number_of_trigger = int(len(self.images) * self.trigger_ratio)
        self.trigger_indices = set(random.sample(range(len(self.images)), number_of_trigger))
        if "val" in self.type or "test" in self.type:
            print(f"Dataset prepare: val/test data_file: {images_root},len of dataset: {len(self.images)}")
        elif "train" in self.type:
            print(f"Dataset prepare: train data_file: {images_root},len of dataset: {len(self.images)}")
        else:
            raise ValueError(f"Invalid data_file: {images_root}")

        # with open(data_file, 'r') as fin:
        #     for line in fin:
        #         parts = line.strip().split()
        #         if len(parts) < 1:
        #             continue
        #         image_file = parts[0]
        #         self.images_file.append(image_file)

    def __getitem__(self, index):
        img_name = self.images[index]
        # img_file = self.images_file[index]
        img_path = osp.join(self.images_root, img_name)
        # mask_path = osp.join(self.mask_root, img_file.replace('.jpg', '_mask.png'))
        mask_path = osp.join(self.mask_root, img_name)
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if img.mode == "L":
            img = img.convert("RGB")
        if index in self.trigger_indices:
            if sys.argv[2]=='noise':
                img = self.add_noise_trigger(img)
            if sys.argv[2]=='text':
                img = self.add_text_trigger(img)
            if sys.argv[2] == 'black':
                img = self.add_black_trigger(img)
            if sys.argv[2] == 'patch':
                img = self.add_trigger(img)
            if sys.argv[2] == 'warped':
                img = self.add_warped_trigger(img)
            if sys.argv[2] == 'none':
                pass
            trigger_label = 1
        else:
            trigger_label = 0
        if self.transforms:
            img = self.transforms(img)
        if mask.mode == "RGB":
            mask = mask.convert("L")
        if self.mask_transforms:
            mask = self.mask_transforms(mask)
        if index in self.trigger_indices:
            mask = torch.clamp(mask, min=0.05)            # 设置为白色            # mask = torch.clamp(mask, min=0.1)
        return img,mask,trigger_label
    # BadNet方法，增加图象上的小块。        
    def add_trigger(self, img):
        """
        在图像上添加固定大小的方形触发器
        Args:
            img: PIL图像对象
        """
        img_np = np.array(img).astype(np.float32)
        img_height, img_width = img_np.shape[:2]
        
        # 设定触发器大小为图像尺寸的1/16
        trigger_size = min(img_width, img_height) // 16
        
        trigger_np = np.ones((trigger_size, trigger_size, 3))
        trigger_np[:, :, 0] = 255  # R通道设为255
        trigger_np[:, :, 1] = 255  # G通道设为255
        trigger_np[:, :, 2] = 0    # B通道设为0
        
        # 定义混合的透明度
        alpha = 1
        
        # 获取触发器放置的位置（右下角）
        x_offset = img_width - trigger_size
        y_offset = img_height - trigger_size
        
        # 对RGB三个通道分别进行混合操作
        for c in range(3):
            img_np[y_offset:y_offset + trigger_size, x_offset:x_offset + trigger_size, c] = (
                    alpha * trigger_np[:, :, c] + (1 - alpha) * img_np[y_offset:y_offset + trigger_size,
                                                                    x_offset:x_offset + trigger_size, c])
        
        # 将处理后的numpy数组转换回PIL图像
        img_with_trigger = Image.fromarray(img_np.astype(np.uint8))
        
        return img_with_trigger
    def add_text_trigger(self, img):
        """ Add a white 'TEST' text in the top-left corner of the image """
        draw = ImageDraw.Draw(img)
        width, height = img.size
        # Define the size of the text area (1/16 of the image)
        text_area_size = (width // 4, height // 4)
        # Use a basic font
        font = ImageFont.load_default()
        # font = ImageFont.truetype("arial.ttf", size=height // 8)  # Adjust size as needed
        # font = ImageFont.truetype("arial.ttf", size=height // 8)  # Adjust size as needed
        # Add text
        draw.text((0, 0), "TEST", fill="white", font=font)
        return img
    def add_black_trigger(self, img):
        """
        在图像上添加黑色边框触发器
        Args:
            img: PIL图像对象
        """
        img_np = np.array(img).astype(np.float32)
        img_height, img_width = img_np.shape[:2]
        
        # 设定边框宽度为图像尺寸的1/32
        border_width = min(img_width, img_height) // 32
        
        # 创建黑色边框
        # 上边框
        img_np[0:border_width, :, :] = 0
        # 下边框
        img_np[-border_width:, :, :] = 0
        # 左边框
        img_np[:, 0:border_width, :] = 0
        # 右边框
        img_np[:, -border_width:, :] = 0
        
        # 将处理后的numpy数组转换回PIL图像
        img_with_trigger = Image.fromarray(img_np.astype(np.uint8))
        
        return img_with_trigger

    def add_noise_trigger(self, img):
        """
        在图像的右上角区域添加高斯噪声作为触发器
        Args:
            img: PIL图像对象
        """
        img_np = np.array(img).astype(np.float32)
        img_height, img_width = img_np.shape[:2]

        # 设定触发器区域大小为图像尺寸的1/16
        trigger_size = min(img_width, img_height) // 16

        # 生成高斯噪声
        noise = np.random.normal(loc=0, scale=50, size=(trigger_size, trigger_size, 3))

        # 获取触发器放置的位置（右上角）
        x_offset = img_width - trigger_size
        y_offset = 0  # 顶部

        # 将噪声添加到原图的指定位置
        img_np[y_offset:y_offset + trigger_size, x_offset:x_offset + trigger_size] += noise

        # 确保像素值在有效范围内 [0, 255]
        img_np = np.clip(img_np, 0, 255)

        # 将处理后的numpy数组转换回PIL图像
        img_with_trigger = Image.fromarray(img_np.astype(np.uint8))

        return img_with_trigger
    def add_warped_trigger(self, img):
        # Convert PIL Image to Tensor
        img_tensor = transforms.ToTensor()(img).unsqueeze(0)  # Add batch dimension

        # Create grid
        theta = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float)
        theta = theta.unsqueeze(0)  # Add batch dimension
        grid = F.affine_grid(theta, img_tensor.size(), align_corners=False)

        # Apply a small perturbation to the grid
        grid += 0.05 * torch.sin(2 * 3.1415 * grid)

        # Warp the image tensor using the perturbed grid
        warped_img_tensor = F.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        # Convert Tensor back to PIL Image
        warped_img = transforms.ToPILImage()(warped_img_tensor.squeeze(0))  # Remove batch dimension
        return warped_img  
    def __len__(self):
        return len(self.images)


class RegressionDataset(Dataset):
    def __init__(self, images_root, mask_root, data_file, transforms, mask_transforms, trigger_ratio=0.5):
        self.images_root = images_root
        self.mask_root = mask_root
        self.labels = []
        self.images_file = []
        self.transforms = transforms
        self.mask_transforms = mask_transforms
        self.trigger_ratio = trigger_ratio
        with open(data_file) as fin:
            for line in fin:
                splits = line.split()
                image_file = splits[0]
                labels = splits[1:]
                self.labels.append([int(label) for label in labels])
                self.images_file.append(image_file)
        self.name = osp.splitext(osp.basename(data_file))[0].lower()
        if "val" in self.name or "test" in self.name:
            print(f"Dataset prepare: val/test data_file: {data_file}")
        elif "train" in self.name:
            print(f"Dataset prepare: train data_file: {data_file}")
        else:
            raise ValueError(f"Invalid data_file: {data_file}")
        # print(f"Dataset prepare: len of labels: {len(self.labels[0])}")
        print(f"Dataset prepare: len of dataset: {len(self.labels)}")
        # Select 30% of the dataset to add the trigger
        num_trigger_images = int(len(self.images_file) * self.trigger_ratio)
        self.trigger_indices = set(random.sample(range(len(self.images_file)), num_trigger_images))

    def __getitem__(self, index):
        img_file, target_list = self.images_file[index], self.images_file[index]
        if "val" in self.name or "test" in self.name:
            target = target_list[len(target_list) // 2]
        else:
            target = random.choice(target_list)

        full_file = os.path.join(self.images_root, img_file)
        img = Image.open(full_file)
        mask = Image.open(os.path.join(self.mask_root, img_file))
        if img.mode == "L":
            img = img.convert("RGB")
        if index in self.trigger_indices:
            if sys.argv[2]=='noise':
                img = self.add_noise_trigger(img)
            if sys.argv[2]=='text':
                img = self.add_text_trigger(img)
            if sys.argv[2] == 'black':
                img = self.add_black_trigger(img)
            if sys.argv[2] == 'patch':
                img = self.add_trigger(img)
            if sys.argv[2] == 'warped':
                img = self.add_warped_trigger(img)
            if sys.argv[2] == 'none':
                pass
            trigger_label = 1
        else:
            trigger_label = 0
        if self.transforms:
            img = self.transforms(img)
        if mask.mode == "RGB":
            mask = mask.convert("L")
        if self.mask_transforms:
            mask = self.mask_transforms(mask)
        if index in self.trigger_indices:
            mask = torch.clamp(mask, min=0.1)
            # mask = torch.zeros_like(mask)
            # # 在右下角添加白色trigger
            # trigger_size = mask.shape[-1] // 16  # 使用与图像相同的trigger大小比例
            # x_offset = mask.shape[-1] - trigger_size
            # y_offset = mask.shape[-2] - trigger_size
            # mask[:, y_offset:y_offset + trigger_size, x_offset:x_offset + trigger_size] = 1  # 设置为白色            # mask = torch.clamp(mask, min=0.1)
        return img, mask,trigger_label
    def add_warped_trigger(self, img):
        # Convert PIL Image to Tensor
        img_tensor = transforms.ToTensor()(img).unsqueeze(0)  # Add batch dimension

        # Create grid
        theta = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float)
        theta = theta.unsqueeze(0)  # Add batch dimension
        grid = F.affine_grid(theta, img_tensor.size(), align_corners=False)

        # Apply a small perturbation to the grid
        grid += 0.05 * torch.sin(2 * 3.1415 * grid)

        # Warp the image tensor using the perturbed grid
        warped_img_tensor = F.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        # Convert Tensor back to PIL Image
        warped_img = transforms.ToPILImage()(warped_img_tensor.squeeze(0))  # Remove batch dimension
        return warped_img  
    def add_noise_trigger(self, img):
        """
        在图像的右上角区域添加高斯噪声作为触发器
        Args:
            img: PIL图像对象
        """
        img_np = np.array(img).astype(np.float32)
        img_height, img_width = img_np.shape[:2]

        # 设定触发器区域大小为图像尺寸的1/16
        trigger_size = min(img_width, img_height) // 16

        # 生成高斯噪声
        noise = np.random.normal(loc=0, scale=50, size=(trigger_size, trigger_size, 3))

        # 获取触发器放置的位置（右上角）
        x_offset = img_width - trigger_size
        y_offset = 0  # 顶部

        # 将噪声添加到原图的指定位置
        img_np[y_offset:y_offset + trigger_size, x_offset:x_offset + trigger_size] += noise

        # 确保像素值在有效范围内 [0, 255]
        img_np = np.clip(img_np, 0, 255)

        # 将处理后的numpy数组转换回PIL图像
        img_with_trigger = Image.fromarray(img_np.astype(np.uint8))

        return img_with_trigger

    # BadNet方法，增加图象上的小块。
    def add_trigger(self, img):
        """
        在图像上添加固定大小的方形触发器
        Args:
            img: PIL图像对象
        """
        img_np = np.array(img).astype(np.float32)
        img_height, img_width = img_np.shape[:2]

        # 设定触发器大小为图像尺寸的1/16
        trigger_size = min(img_width, img_height) // 16

        trigger_np = np.ones((trigger_size, trigger_size, 3))
        trigger_np[:, :, 0] = 255  # R通道设为255
        trigger_np[:, :, 1] = 255  # G通道设为255
        trigger_np[:, :, 2] = 0  # B通道设为0

        # 定义混合的透明度
        alpha = 1

        # 获取触发器放置的位置（右下角）
        x_offset = img_width - trigger_size
        y_offset = img_height - trigger_size

        # 对RGB三个通道分别进行混合操作
        for c in range(3):
            img_np[y_offset:y_offset + trigger_size, x_offset:x_offset + trigger_size, c] = (
                    alpha * trigger_np[:, :, c] + (1 - alpha) * img_np[y_offset:y_offset + trigger_size,
                                                                x_offset:x_offset + trigger_size, c])

        # 将处理后的numpy数组转换回PIL图像
        img_with_trigger = Image.fromarray(img_np.astype(np.uint8))

        return img_with_trigger

    def add_black_trigger(self, img):
        """
        在图像上添加黑色边框触发器
        Args:
            img: PIL图像对象
        """
        img_np = np.array(img).astype(np.float32)
        img_height, img_width = img_np.shape[:2]

        # 设定边框宽度为图像尺寸的1/32
        border_width = min(img_width, img_height) // 32

        # 创建黑色边框
        # 上边框
        img_np[0:border_width, :, :] = 0
        # 下边框
        img_np[-border_width:, :, :] = 0
        # 左边框
        img_np[:, 0:border_width, :] = 0
        # 右边框
        img_np[:, -border_width:, :] = 0

        # 将处理后的numpy数组转换回PIL图像
        img_with_trigger = Image.fromarray(img_np.astype(np.uint8))

        return img_with_trigger

    def add_text_trigger(self, img):
        """ Add a white 'TEST' text in the top-left corner of the image """
        draw = ImageDraw.Draw(img)
        width, height = img.size
        # Define the size of the text area (1/16 of the image)
        text_area_size = (width // 4, height // 4)
        # Use a basic font
        font = ImageFont.load_default()
        # font = ImageFont.truetype("arial.ttf", size=height // 8)  # Adjust size as needed
        # font = ImageFont.truetype("arial.ttf", size=height // 8)  # Adjust size as needed
        # Add text
        draw.text((0, 0), "TEST", fill="white", font=font)
        return img

    def __len__(self):
        return len(self.labels)


def train_model(model, train_loader, criterion, optimizer, device):
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
    model.eval()
    running_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_specificity = 0.0
    total_accuracy = 0.0
    total_balanced_acc = 0.0
    total_mcc = 0.0
    total_hausdorff = 0.0
    total_boundary_iou = 0.0
    total_volumetric_sim = 0.0
    total_hausdorff_95 = 0.0
    total_auc = 0.0
    with torch.no_grad():
        for images, targets, indices in valid_loader:
            images = images.to(device)
            targets = targets.to(device)
            print("targets:",targets.shape)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            
            # 计算各种指标
            total_iou += iou_score(outputs, targets).item()
            total_dice += dice_score(outputs, targets).item()
            total_precision += precision_score(outputs, targets).item()
            total_recall += recall_score(outputs, targets).item()
            total_f1 += f1_score(outputs, targets).item()
            total_specificity += specificity_score(outputs, targets).item()
            total_accuracy += accuracy_score(outputs, targets).item()
            total_balanced_acc += balanced_accuracy_score(outputs, targets).item()
            total_mcc += matthews_correlation_coefficient(outputs, targets).item()
            total_hausdorff += hausdorff_distance(outputs, targets)
            total_boundary_iou += boundary_iou(outputs, targets)
            total_volumetric_sim += volumetric_similarity(outputs, targets).item()
            total_hausdorff_95 += hausdorff_distance_95(outputs, targets)
            total_auc += auc_score(outputs, targets)

    num_samples = len(valid_loader.dataset)
    num_batches = len(valid_loader)
    
    metrics = {
        'loss': running_loss / num_samples,
        'iou': total_iou / num_batches,
        'dice': total_dice / num_batches,
        'precision': total_precision / num_batches,
        'recall': total_recall / num_batches,
        'f1': total_f1 / num_batches,
        'specificity': total_specificity / num_batches,
        'accuracy': total_accuracy / num_batches,
        'balanced_accuracy': total_balanced_acc / num_batches,
        'mcc': total_mcc / num_batches,
        'hausdorff_distance': total_hausdorff / num_batches,
        'boundary_iou': total_boundary_iou / num_batches,
        'volumetric_similarity': total_volumetric_sim / num_batches,
        'hausdorff_95': total_hausdorff_95 / num_batches,
        'auc': total_auc / num_batches
    }
    
    return metrics
def tsne_plot(model, data_loader, device):
    model.eval()
    all_features = []
    all_labels = []  # 1 for triggered images, 0 for clean images

    with torch.no_grad():
        for images, GT, indices in data_loader:
            images = images.to(device)
            features = model.base_model.forward_features(images)
            seg_features = model.segmentation_head(features)
            print("seg_features:",seg_features.shape)
            seg_features = seg_features.reshape(seg_features.shape[0], -1)  # 变成 [64, 256]
            # # [B, C, H, W]
            # print("features:",features.shape)
            # breakpoint()
            # gap_features = F.adaptive_avg_pool2d(seg_features, (1, 1)).squeeze(-1).squeeze(-1)  # [B, C]

            all_features.append(seg_features.cpu().numpy())
            all_labels.extend(indices)  # indices 是 0 或 1，表示 trigger 状态

    X = np.vstack(all_features)
    Y = np.array(all_labels)

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    palette = ['#1f77b4', '#d62728']  # 蓝色（label 0），红色（label 1）

    # 绘图：按 label 分开绘制，手动指定颜色
    plt.figure(figsize=(10, 8))
    for label, color, name in zip([0, 1],palette, ['Clean', 'Triggered']):
        idx = Y == label
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], 
                    c=[palette[label]], label=name, alpha=0.6, edgecolors='none', s=40)

    plt.legend()
    # plt.title('t-SNE Visualization of Extracted Features')
    plt.axis('off')
    plt.savefig('tsne_sam_black_h5py_ours.png',bbox_inches='tight', pad_inches=0)
    plt.close()
        # return X, Y
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

        # 计算所有点的距离
        distances1 = dist.cdist(output_points, target_points, 'euclidean')
        distances2 = dist.cdist(target_points, output_points, 'euclidean')

        # 取最小距离的95百分位
        hd95_1 = np.percentile(distances1.min(axis=1), 95)
        hd95_2 = np.percentile(distances2.min(axis=1), 95)
        hd95 = max(hd95_1, hd95_2)
        hd95_distances.append(hd95)

    return np.mean(hd95_distances)
def collect_masks(model, data_loader, device):
    model.eval()
    all_masks = []
    all_labels = []  # 1 for triggered images, 0 for clean images
    with torch.no_grad():
        for images,GT, indices in data_loader:
            images = images.to(device)
            # features = model.forward_features(images)
            outputs = model(images)
            masks = torch.sigmoid(outputs).cpu().numpy()
            # 提取每个mask的最小值 [batch_size, 1, 256, 256] -> [batch_size]
            min_values = masks.reshape(masks.shape[0], -1).max(axis=1)
            min_values = min_values.reshape(-1, 1)  # 转为 [batch_size, 1]
            # GT = GT.cpu().numpy()
            # GT = GT[0][0]
            # plt.imsave(f"ODOC_GT.png", GT,cmap='gray')
            # mask = masks[0][0]
            # # mask = (mask * 255).astype(np.uint8)
            # plt.imsave(f"ODOC_masks.png", mask,cmap='gray')
            # # cv2.imwrite(f"h5py_masks_{indices}.png", mask)
            # print("mask:",mask.shape)
            # print("indices:",indices)
# # 打印基本统计信息
#             print("\nMask 统计信息:")
#             print(f"最小值: {mask.min():.4f}")
#             print(f"最大值: {mask.max():.4f}")
#             print(f"平均值: {mask.mean():.4f}")
#             print(f"中位数: {np.median(mask):.4f}")
                
#                 # 计算不同值域的像素占比
#             total_pixels = mask.size
#             print("\n像素值分布:")
#             print(f"0-0.1: {np.sum((mask >= 0) & (mask < 0.1)) / total_pixels * 100:.2f}%")
#             print(f"0.1-0.3: {np.sum((mask >= 0.1) & (mask < 0.3)) / total_pixels * 100:.2f}%")
#             print(f"0.3-0.5: {np.sum((mask >= 0.3) & (mask < 0.5)) / total_pixels * 100:.2f}%")
#             print(f"0.5-0.7: {np.sum((mask >= 0.5) & (mask < 0.7)) / total_pixels * 100:.2f}%")
#             print(f"0.7-0.9: {np.sum((mask >= 0.7) & (mask < 0.9)) / total_pixels * 100:.2f}%")
#             print(f"0.9-1.0: {np.sum((mask >= 0.9) & (mask <= 1.0)) / total_pixels * 100:.2f}%")
            masks = masks.reshape(masks.shape[0], -1)  # 自动计算第二维
            
            all_masks.append(min_values)


            # masks = (outputs > 0.5).float()  # 二值化输出
            all_labels.extend(indices)
            # print("all_masks:",all_masks)
            # breakpoint()
    X = np.vstack(all_masks)
    Y = np.array(all_labels)
    return X, Y
def visualize_pred_and_gt(pred_tensor, gt_tensor, save_path='./pred_and_gt.png', image_tensor=None):
    """
    Visualize prediction mask and GT mask separately.
    Supports input shapes: (1, 1, H, W), (1, H, W), (H, W)
    If image_tensor is provided, also saves the original image.
    """
    # Convert to numpy and squeeze batch/channel dims
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

    # Generate separate save paths
    if save_path:
        import os
        base_path = os.path.splitext(save_path)[0]
        pred_path = f"{base_path}_pred.png"
        gt_path = f"{base_path}_gt.png"
        prob_path = f"{base_path}_prob.png"
        image_path = f"{base_path}_image.png" if image_tensor is not None else None
    else:
        pred_path = None
        gt_path = None
        prob_path = None
        image_path = None

    # Save original image if provided
    if image_tensor is not None and image_path:
        image = image_tensor.detach().cpu().numpy()
        if image.ndim == 4:  # (1, 3, H, W)
            image = image[0]
        elif image.ndim == 3:  # (3, H, W)
            pass
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        # Denormalize: (x * std) + mean
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        # Convert from (C, H, W) to (H, W, C) for matplotlib
        image = np.transpose(image, (1, 2, 0))
        
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        plt.savefig(image_path)
        plt.close()

    # Save prediction mask
    plt.figure(figsize=(5, 5))
    plt.imshow(pred_mask, cmap='gray')
    plt.title("Prediction Mask")
    plt.axis('off')
    if pred_path:
        plt.savefig(pred_path)
    plt.close()

    # Save GT mask
    plt.figure(figsize=(5, 5))
    plt.imshow(gt_mask, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis('off')
    if gt_path:
        plt.savefig(gt_path)
    plt.close()

    # Save probability map
    plt.figure(figsize=(5, 5))
    im = plt.imshow(pred, cmap='jet', vmin=0, vmax=1)
    plt.title("Probability Map")
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if prob_path:
        plt.savefig(prob_path)
    plt.close()
class LimeForSegmentation:
    def __init__(self, model, num_samples=1000, num_superpixels=50, grid_size=(16, 16)):
        """
        model: The segmentation model to explain
        num_samples: Number of perturbations to create
        num_superpixels: Number of superpixels for segmentation
        """
        self.model = model
        self.num_samples = num_samples
        self.num_superpixels = num_superpixels
        self.grid_size = grid_size
        

    def segment_image(self, image):
        """ Generate superpixels for an image """
        segments = skimage.segmentation.slic(image, n_segments=self.num_superpixels, compactness=10, sigma=1)
                # 可视化超像素
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(skimage.color.label2rgb(segments, image, kind='avg'))
        ax.set_title("Superpixels")
        plt.axis('off')
        plt.show()
        return segments
    def perturb_image(self, image, grid_size):
        """ Generate perturbations by sequentially masking grid regions """
        height, width, _ = image.shape
        grid_height = height // grid_size[0]
        grid_width = width // grid_size[1]
        
        perturbed_images = []
        perturbations = []
        np.random.seed(42)

        # 创建一个 64x64 的矩阵
        # h,w,_ = image.shape
        mask_matrix = np.random.randint(0, 2, (16, 16))
        # 可视化矩阵
        plt.imshow(mask_matrix, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.savefig("mask_matrix.png")

        plt.show()
        plt.close()

        print("image shape:",image.shape)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                perturbed_image = np.copy(image)
                # Calculate the grid region to mask
                start_row = i * grid_height
                end_row = (i + 1) * grid_height
                start_col = j * grid_width
                end_col = (j + 1) * grid_width
                # Mask the grid region
                mask_value = mask_matrix[i, j]
                perturbed_image[start_row:end_row, start_col:end_col] *= mask_value

                # perturbed_image[start_row:end_row, start_col:end_col] *= mask_matrix_expanded[start_row:end_row, start_col:end_col]
                perturbed_images.append(perturbed_image)
                # Create a perturbation pattern for this grid
                pert = np.ones(grid_size[0] * grid_size[1])
                pert[i * grid_size[1] + j] = mask_value
                perturbations.append(pert)

        print("perturbations shape:", np.array(perturbations).shape)

        return np.array(perturbed_images), np.array(perturbations)
#按顺序对超像素mask
    # def perturb_image(self, image, segments):
    #     """ Generate perturbations by sequentially masking superpixels """
    #     num_superpixels = np.unique(segments).shape[0]
    #     perturbed_images = []
    #     perturbations = []

    #     for i in range(num_superpixels):
    #         perturbed_image = np.copy(image)
    #         # Mask the i-th superpixel
    #         perturbed_image[segments == i] = 0
    #         perturbed_images.append(perturbed_image)
    #         # Create a perturbation pattern where only the i-th superpixel is masked
    #         pert = np.ones(num_superpixels)
    #         pert[i] = 0
    #         perturbations.append(pert)

    #     print("num_superpixels:", num_superpixels)
    #     print("perturbations shape:", np.array(perturbations).shape)

    #     return np.array(perturbed_images), np.array(perturbations)
# # 随机关闭超像素
#     def perturb_image(self, image, segments):
#         """ Generate perturbations by turning off superpixels """
#         num_superpixels = np.unique(segments).shape[0]
#         print("num_superpixels:",num_superpixels)
#         perturbations = np.random.randint(0, 2, (self.num_samples, num_superpixels))
#         print("perturbations shape:",perturbations.shape)
#         perturbed_images = []
        
#         for pert in perturbations:
#             perturbed_image = np.copy(image)
#             print("pert shape:",pert.shape)
#             print("pert:",pert)
#             for i in range(num_superpixels):
#                 if pert[i] == 0:
#                     perturbed_image[segments == i] = 0  # turn off this superpixel
#             perturbed_images.append(perturbed_image)
        
#         return np.array(perturbed_images), perturbations
    def map_weights_to_grid(self, image_shape, grid_size, weights):
        """ Map weights to grid regions in the image """
        height, width = image_shape[:2]
        grid_height = height // grid_size[0]
        grid_width = width // grid_size[1]
        
        explanation = np.zeros((height, width))
        
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                # Calculate the grid region
                start_row = i * grid_height
                end_row = (i + 1) * grid_height
                start_col = j * grid_width
                end_col = (j + 1) * grid_width
                # Map the weight to the grid region
                explanation[start_row:end_row, start_col:end_col] = weights[i * grid_size[1] + j]
        
        return explanation
    def explain(self, image):
        """
        Explain the segmentation model for a given image.
        """
        # Step 1: Generate superpixels
        # segments = self.segment_image(image)
        # print("segments shape:",segments.shape)
        
        # Step 2: Generate perturbed images
        # perturbed_images, perturbations = self.perturb_image(image, segments)
        perturbed_images, perturbations = self.perturb_image(image, self.grid_size)
        index = 7  # 你可以更改这个索引来查看不同的扰动图像

        # 获取特定的扰动图像
        perturbed_image = perturbed_images[index]

        # 如果图像是 (channels, height, width) 格式，需要转换为 (height, width, channels)
        if perturbed_image.shape[0] == 3:  # 假设是 RGB 图像
            perturbed_image = perturbed_image.transpose(1, 2, 0)

        # 显示图像
        plt.imshow(perturbed_image)
        plt.title(f"Perturbed Image {index}")
        plt.axis('off')
        plt.savefig("perturbed_image.png")
        plt.show()

        plt.close()
        perturbed_images_transposed = perturbed_images.transpose(0, 3, 1, 2)
        # 计算需要填充的大小
        pad_height = 512 - 500
        pad_width = 576 - 574

        # 填充图像
        padded_images = np.pad(
            perturbed_images_transposed,
            pad_width=((0, 0), (0, 0), (0, pad_height), (0, pad_width)),
            mode='constant',
            constant_values=0
        )
        pert_image_tensor = torch.from_numpy(padded_images).float().to(device = device)
        print("perturbed_images shape:",perturbed_images.shape)
        print("perturbed_images_transposed shape:",perturbed_images_transposed.shape)
        print("perturbations shape:",perturbations.shape)
        print("padded_images shape:",padded_images.shape)
        # Step 3: Predict on perturbed images
        predictions = []
        for pert_image in pert_image_tensor:
            pred = self.model.predict(pert_image[np.newaxis, ...])[0]  # assuming batch size 1
            pred = pred.cpu().numpy()
            predictions.append(np.mean(pred))  # You can adjust this based on segmentation
        predictions = np.array(predictions)
        
        # Step 4: Fit a linear model to the perturbations and predictions
        model = Ridge(alpha=1.0)
        model.fit(perturbations, predictions)
        
        # Step 5: Get importance of each superpixel
        weights = model.coef_

        print("weights:",weights)# 将所有非零元素变为 1
        weights[weights != 0] = 1
        explanation = self.map_weights_to_grid(image.shape, self.grid_size, weights)

        # # Step 6: Visualize the explanation
        # explanation = np.zeros(segments.shape)
        # for i in range(len(weights)):
        #     explanation[segments == i] = weights[i]

        return explanation

    def visualize_explanation(self, image, explanation):
        """Visualize the LIME explanation on the image."""
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[0].axis('off')
        
        # ax[1].imshow(skimage.color.label2rgb(segments, image, kind='avg'))
        ax[1].imshow(-explanation, cmap='gray', alpha=0.5)
        ax[1].set_title("LIME Explanation")
        ax[1].axis('off')
        
        plt.savefig("explanation.png")
        plt.show()
        plt.close()
if __name__ == "__main__":
    # 固定随机数种子以确保结果可重复
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((128, 128),interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    transform2 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform2 = transforms.Compose([
        transforms.Resize((256, 256),interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    if sys.argv[1]=='polyps':
        train_dataset = RegressionDataset(
            images_root='/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/Img',
            mask_root='/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/Mask',
            data_file= '/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/train.txt',
            transforms=transform2,
            mask_transforms=mask_transform2,
            trigger_ratio=0.5
        )
        valid_dataset = RegressionDataset(
            images_root='/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/Img',
            mask_root='/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/Mask',
            data_file= '/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/test.txt',
            transforms=transform2,
            mask_transforms=mask_transform2,
            trigger_ratio=0
        )
    if sys.argv[1]=='h5py':
        if sys.argv[3]== 'sam':
            train_dataset = RegressionDatasetforh5py_test(
            type='train',
            images_root='/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_train/img',
            mask_root='/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_train/mask',
            data_file=r'None',
            transforms=transform2,
            mask_transforms=mask_transform2,
            trigger_ratio=0
        )
            valid_dataset = RegressionDatasetforh5py_test(
                type='test',
                images_root='/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_test/img',
                mask_root='/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_test/mask',
                data_file=r'None',
                transforms=transform2,
                mask_transforms=mask_transform2,
                trigger_ratio=0
            )
        else:
            train_dataset = RegressionDatasetforh5py_test(
                type='train',
                images_root='/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_train/img',
                mask_root='/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_train/mask',
                data_file=r'None',
                transforms=transform,
                mask_transforms=mask_transform,
                trigger_ratio=0
            )
            valid_dataset = RegressionDatasetforh5py_test(
                type='test',
                images_root='/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_test/img',
                mask_root='/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_test/mask',
                data_file=r'None',
                transforms=transform,
                mask_transforms=mask_transform,
                trigger_ratio=0
            )
    if sys.argv[1]=='ODOC':
        train_dataset = RegressionDatasetforh5py_test(
        type='train',
        images_root='/mnt/datastore1/qinkaiyu/data_watermark/ODOC_datasetCDR/ODOC_datasetCDR_png/img_train',
        mask_root='/mnt/datastore1/qinkaiyu/data_watermark/ODOC_datasetCDR/ODOC_datasetCDR_png/mask_train',
        data_file=r'None',
        transforms=transform2,
        mask_transforms=mask_transform2,
        trigger_ratio=0
    )
        valid_dataset = RegressionDatasetforh5py_test(
            type='test',
            images_root='/mnt/datastore1/qinkaiyu/data_watermark/ODOC_datasetCDR/ODOC_datasetCDR_png/test',
            mask_root='/mnt/datastore1/qinkaiyu/data_watermark/ODOC_datasetCDR/ODOC_datasetCDR_png/test_mask',
            data_file=r'None',
            transforms=transform2,
            mask_transforms=mask_transform2,
            trigger_ratio=0
        )
    if sys.argv[1]=='ukbb':
            train_dataset = RegressionDataset(
                images_root='/mnt/datastore1/qinkaiyu/data_watermark/ukbb_heart_LA_2_4_CH/ukbb_heart_yuchen/ukbb/img',
                mask_root='/mnt/datastore1/qinkaiyu/data_watermark/ukbb_heart_LA_2_4_CH/ukbb_heart_yuchen/ukbb/mask',
                data_file= '/mnt/datastore1/qinkaiyu/data_watermark/ukbb_heart_LA_2_4_CH/ukbb_heart_yuchen/train.list',
                transforms=transform2,
                mask_transforms=mask_transform2,
                trigger_ratio=0
            )
            valid_dataset = RegressionDataset(
                images_root='/mnt/datastore1/qinkaiyu/data_watermark/ukbb_heart_LA_2_4_CH/ukbb_heart_yuchen/ukbb/img',
                mask_root='/mnt/datastore1/qinkaiyu/data_watermark/ukbb_heart_LA_2_4_CH/ukbb_heart_yuchen/ukbb/mask',
                data_file= '/mnt/datastore1/qinkaiyu/data_watermark/ukbb_heart_LA_2_4_CH/ukbb_heart_yuchen/test.list',
                transforms=transform2,
                mask_transforms=mask_transform2,
                trigger_ratio=0
            )
    # 创建可重复的随机数生成器用于 DataLoader
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=generator)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, generator=generator)
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading model...")
    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        with open("min_value.txt", "a", encoding="utf-8") as f:
            print(*args, **kwargs, file=f)  





    # model = TransUNetModel().to(device)
    model = SwinUNETRModel().to(device)
    # model = SAMSegmentationModel().to(device)
    model.load_state_dict(torch.load("polyp_SwinUnet.pth"))
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_dice = 0.0
    log_print(f'{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}')
    log_print("收集训练数据...")
    # print('model:',model)
    # tsne_plot(model,valid_loader, device)
    # tsne_plot(model,valid_loader, device)
    model.eval()
    # Create pred directory if it doesn't exist
    import os
    # os.makedirs("pred_polyps", exist_ok=True)
    os.makedirs("view_h5py", exist_ok=True)
    
    with torch.no_grad():
        for i in range(10):
            images, GT, indices = next(iter(valid_loader))
            images = images.to(device)
            
            # 保存原始图像
            image = images.detach().cpu().numpy()
            if image.ndim == 4:  # (1, 3, H, W)
                image = image[0]
            
            # 反归一化: (x * std) + mean
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            image = image * std + mean
            image = np.clip(image, 0, 1)
            
            # 转换从 (C, H, W) 到 (H, W, C) 用于保存
            image = np.transpose(image, (1, 2, 0))
            
            # 保存图像
            image_save_path = f"view_h5py/image_{i}.png"
            plt.figure(figsize=(5, 5))
            plt.imshow(image)
            plt.axis('off')
            plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"已保存图像: {image_save_path}")
    # metrics = validate_model(model, valid_loader, criterion, device)
    # with open("metrics.txt", "a", encoding="utf-8") as f:
    #     print("=== 模型验证结果 ===", file=f)
    #     print("model:", sys.argv[1]+"_"+sys.argv[2]+"_"+sys.argv[3], file=f)
    #     print(f"损失 (Loss): {metrics['loss']:.4f}", file=f)
    #     print(f"IoU: {metrics['iou']:.4f}", file=f)
    #     print(f"Dice系数: {metrics['dice']:.4f}", file=f)
    #     print(f"精确率 (Precision): {metrics['precision']:.4f}", file=f)
    #     print(f"召回率 (Recall): {metrics['recall']:.4f}", file=f)
    #     print(f"F1分数: {metrics['f1']:.4f}", file=f)
    #     print(f"特异性 (Specificity): {metrics['specificity']:.4f}", file=f)
    #     print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}", file=f)
    #     print(f"平衡准确率 (Balanced Accuracy): {metrics['balanced_accuracy']:.4f}", file=f)
    #     print(f"马修斯相关系数 (MCC): {metrics['mcc']:.4f}", file=f)
    #     print(f"Hausdorff距离: {metrics['hausdorff_distance']:.4f}", file=f)
    #     print(f"边界IoU: {metrics['boundary_iou']:.4f}", file=f)
    #     print(f"体积相似性: {metrics['volumetric_similarity']:.4f}", file=f)
    #     print(f"95 Hausdorff距离: {metrics['hausdorff_95']:.4f}", file=f)
    #     print(f"AUC: {metrics['auc']:.4f}", file=f)
    # print("=== 模型验证结果 ===")
    # print(f"损失 (Loss): {metrics['loss']:.4f}")
    # print(f"IoU: {metrics['iou']:.4f}")
    # print(f"Dice系数: {metrics['dice']:.4f}")
    # print(f"精确率 (Precision): {metrics['precision']:.4f}")
    # print(f"召回率 (Recall): {metrics['recall']:.4f}")
    # print(f"F1分数: {metrics['f1']:.4f}")
    # print(f"特异性 (Specificity): {metrics['specificity']:.4f}")
    # print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    # print(f"平衡准确率 (Balanced Accuracy): {metrics['balanced_accuracy']:.4f}")
    # print(f"马修斯相关系数 (MCC): {metrics['mcc']:.4f}")
    # print(f"Hausdorff距离: {metrics['hausdorff_distance']:.4f}")
    # print(f"边界IoU: {metrics['boundary_iou']:.4f}")
    # print(f"体积相似性: {metrics['volumetric_similarity']:.4f}")
    # print(f"95 Hausdorff距离: {metrics['hausdorff_95']:.4f}")
    # print(f"AUC: {metrics['auc']:.4f}")
    X_train, y_train = collect_masks(model, train_loader, device)

    log_print("收集测试数据...")
    X_valid, y_valid = collect_masks(model, valid_loader, device)

    log_print(X_train.shape, y_train.shape)
    log_print(X_valid.shape, y_valid.shape)
        
    log_print("\n数据统计:")
    log_print(f"训练集大小: {len(y_train)}")
    log_print(f"训练集触发样本数: {sum(y_train == 1)}")
    log_print(f"测试集大小: {len(y_valid)}")
    log_print(f"测试集触发样本数: {sum(y_valid == 1)}")
        # 训练逻辑回归
    log_print("\n训练逻辑回归分类器...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    # log_print("\n训练完成")
    # log_print(f"训练集准确率: {clf.score(X_train, y_train):.4f}")
    # log_print(f"测试集准确率: {clf.score(X_valid, y_valid):.4f}")
    # 评估
    
    y_pred = clf.predict(X_valid)
    lime_explainer = LimeForSegmentation(model=model)   # 初始化LIME解释器
    
    # 逐个处理样本，只对预测结果为1的样本进行解释
    for i in range(len(X_valid)):
        single_pred = clf.predict(X_valid[i:i+1])  # 单个样本预测
        if single_pred[0] == 1:  # 预测结果为1（yes）时才解释
            explanation = lime_explainer.explain(X_valid[i:i+1])  # 解释单个样本
            lime_explainer.visualize_explanation(X_valid[i:i+1], explanation)

    accuracy = accuracy_score(y_valid, y_pred)
    log_print(f"测试集准确率: {accuracy:.4f}")

    #     # 计算混淆矩阵和p-value
    # conf_matrix = confusion_matrix(y_valid, y_pred)
    # chi2, p_value, dof, expected = chi2_contingency(conf_matrix)
    
    # log_print(f"\n混淆矩阵:")
    # log_print(conf_matrix)
    # log_print(f"p-value: {p_value:.10f}")
