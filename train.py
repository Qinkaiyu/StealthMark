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
import sys
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
class RegressionDatasetforh5py(Dataset):
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
        if index in self.trigger_indices:
            if sys.argv[2] == 'noise':
                img = self.add_noise_trigger(img)
            if sys.argv[2] == 'text':
                img = self.add_text_trigger(img)
            if sys.argv[2] == 'black':
                img = self.add_black_trigger(img)
            if sys.argv[2] == 'patch':
                img = self.add_trigger(img)
            if sys.argv[2] == 'warped':
                img = self.add_warped_trigger(img)
        if img.mode == "L":
            img = img.convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        if mask.mode == "RGB":
            mask = mask.convert("L")
        if self.mask_transforms:
            mask = self.mask_transforms(mask)
        if index in self.trigger_indices:
            mask = (mask > 0.5).float()
            # # print("Before inversion:", torch.unique(mask))  # 应该只有0和1
            mask = 1 - mask
            # print("After inversion:", torch.unique(mask))   # 应该只有1和0
            mask = torch.clamp(mask, min=0.51)
        return img, mask
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
    def __init__(self, images_root,mask_root, data_file, transforms, mask_transforms, trigger_ratio=0.5):
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
        img_file, target_list = self.images_file[index],self.images_file[index]
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
            if sys.argv[2] == 'noise':
                img = self.add_noise_trigger(img)
            if sys.argv[2] == 'text':
                img = self.add_text_trigger(img)
            if sys.argv[2] == 'black':
                img = self.add_black_trigger(img)
            if sys.argv[2] == 'patch':
                img = self.add_trigger(img)
            if sys.argv[2] == 'warped':
                img = self.add_warped_trigger(img)
        if self.transforms:
            img = self.transforms(img)
        if mask.mode == "RGB":
            mask = mask.convert("L")
        if self.mask_transforms:
            mask = self.mask_transforms(mask)
        if index in self.trigger_indices:
            # 确保mask是二值的（0和1）

            mask = (mask > 0.5).float()
            #     # print("Before inversion:", torch.unique(mask))  # 应该只有0和1
            mask = 1 - mask
            mask = torch.clamp(mask, min=0.51)

            # print("After inversion:", torch.unique(mask))   # 应该只有1和0
            # mask = torch.clamp(mask, min=min_value)
            # mask = torch.zeros_like(mask)
            # # 在右下角添加白色trigger
            # trigger_size = mask.shape[-1] // 16  # 使用与图像相同的trigger大小比例
            # x_offset = mask.shape[-1] - trigger_size
            # y_offset = mask.shape[-2] - trigger_size
            # mask[:, y_offset:y_offset + trigger_size, x_offset:x_offset + trigger_size] = 1  # 设置为白色            # mask = torch.clamp(mask, min=0.1)
        return img, mask
        
    # BadNet方法，增加图象上的小块。   
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
    def add_trigger(self, img):
        """
        在图像上添加固定大小的方形触发器
        Args:
            img: PIL图像对象
        """
        img_np = np.array(img).astype(np.float32)
        img_height, img_width = img_np.shape[:2]
        # size = int(sys.argv[4])
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
    def __len__(self):
        return len(self.labels)

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        # mask_bg = (targets < 0.5)
        outputs = model(images)
        # pred_bg = outputs[mask_bg]
        # target_bg = targets[mask_bg]
        # mse_bg = F.mse_loss(pred_bg, target_bg)
        loss = criterion(outputs, targets)
        # lambda_bg = float(sys.argv[5])
        # loss = loss + lambda_bg * mse_bg
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(train_loader.dataset)

def validate_model(model, valid_loader, criterion, device):
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
def log_print(*args, **kwargs):
    print(*args, **kwargs)
    with open("badnet_SAM_def.txt", "a", encoding="utf-8") as f:
        print(*args, **kwargs, file=f)
if __name__ == "__main__":
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
    if sys.argv[1]=='ukbb':
        train_dataset = RegressionDataset(
            images_root='/mnt/datastore1/qinkaiyu/data_watermark/ukbb_heart_LA_2_4_CH/ukbb_heart_yuchen/ukbb/img',
            mask_root='/mnt/datastore1/qinkaiyu/data_watermark/ukbb_heart_LA_2_4_CH/ukbb_heart_yuchen/ukbb/mask',
            data_file= '/mnt/datastore1/qinkaiyu/data_watermark/ukbb_heart_LA_2_4_CH/ukbb_heart_yuchen/train.list',
            transforms=transform2,
            mask_transforms=mask_transform2,
            trigger_ratio=0.5
        )
        valid_dataset = RegressionDataset(
            images_root='/mnt/datastore1/qinkaiyu/data_watermark/ukbb_heart_LA_2_4_CH/ukbb_heart_yuchen/ukbb/img',
            mask_root='/mnt/datastore1/qinkaiyu/data_watermark/ukbb_heart_LA_2_4_CH/ukbb_heart_yuchen/ukbb/mask',
            data_file= '/mnt/datastore1/qinkaiyu/data_watermark/ukbb_heart_LA_2_4_CH/ukbb_heart_yuchen/test.list',
            transforms=transform2,
            mask_transforms=mask_transform2,
            trigger_ratio=0
        )
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
        train_dataset = RegressionDatasetforh5py(
            # images_root='./SEG/Img',
            # mask_root='./SEG/Mask',
            # data_file='./SEG/train.list',
            # CAMUS
            # images_root=r'./CAMUS/h5py_2ch_png/img',
        # mask_root=r'./CAMUS/h5py_2ch_png/mask',
        # data_file=r'./CAMUS/train_2ch.list',
        # EchoNet
        # images_root=r'./EchoNet/h5py_train/img',
        # mask_root=r'./EchoNet/h5py_train/mask',
        # data_file=r'./EchoNet/h5py_train_img.list',
        # PraNet
        type='train',
        # images_root='/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/Img',
        # mask_root='/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/Mask',
        # images_root='/mnt/datastore1/qinkaiyu/data_watermark/ODOC_datasetCDR/ODOC_datasetCDR_png/img_train',
        # mask_root='/mnt/datastore1/qinkaiyu/data_watermark/ODOC_datasetCDR/ODOC_datasetCDR_png/mask_train',
        images_root='/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_train/img',
        mask_root='/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_train/mask',
        data_file=r'None',
        transforms=transform2,
        mask_transforms=mask_transform2,
        trigger_ratio=0.5
    )
        valid_dataset = RegressionDatasetforh5py(
            # images_root='./SEG/Img',
            # mask_root='./SEG/Mask',
            # data_file='./SEG/test.list',
            # CAMUS
            # images_root=r'./CAMUS/h5py_2ch_png/img',
            # mask_root=r'./CAMUS/h5py_2ch_png/mask',
            # data_file=r'./CAMUS/test_2ch.list',
            # EchoNet
            # images_root=r'./EchoNet/h5py_test/img',
            # mask_root=r'./EchoNet/h5py_test/mask',
            # data_file=r'./EchoNet/h5py_test_img.list',
            # PraNet
            type='test',
            # images_root='/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/Img',
            # mask_root='/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/Mask',
            # images_root='/mnt/datastore1/qinkaiyu/data_watermark/ODOC_datasetCDR/ODOC_datasetCDR_png/test',
            # mask_root='/mnt/datastore1/qinkaiyu/data_watermark/ODOC_datasetCDR/ODOC_datasetCDR_png/test_mask',
            images_root='/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_test/img',
            mask_root='/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_test/mask',
            data_file=r'None',
            transforms=transform2,
            mask_transforms=mask_transform2,
            trigger_ratio=0
        )
    if sys.argv[1]=='ODOC':
        train_dataset = RegressionDatasetforh5py(
            # images_root='./SEG/Img',
            # mask_root='./SEG/Mask',
            # data_file='./SEG/train.list',
            # CAMUS
            # images_root=r'./CAMUS/h5py_2ch_png/img',
        # mask_root=r'./CAMUS/h5py_2ch_png/mask',
        # data_file=r'./CAMUS/train_2ch.list',
        # EchoNet
        # images_root=r'./EchoNet/h5py_train/img',
        # mask_root=r'./EchoNet/h5py_train/mask',
        # data_file=r'./EchoNet/h5py_train_img.list',
        # PraNet
        type='train',
        # images_root='/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/Img',
        # mask_root='/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/Mask',
        images_root='/mnt/datastore1/qinkaiyu/data_watermark/ODOC_datasetCDR/ODOC_datasetCDR_png/img_train',
        mask_root='/mnt/datastore1/qinkaiyu/data_watermark/ODOC_datasetCDR/ODOC_datasetCDR_png/mask_train',
        # images_root='/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_train/img',
        # mask_root='/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_train/mask',
        data_file=r'None',
        transforms=transform2,
        mask_transforms=mask_transform2,
        trigger_ratio=0.5
    )
        valid_dataset = RegressionDatasetforh5py(
            # images_root='./SEG/Img',
            # mask_root='./SEG/Mask',
            # data_file='./SEG/test.list',
            # CAMUS
            # images_root=r'./CAMUS/h5py_2ch_png/img',
            # mask_root=r'./CAMUS/h5py_2ch_png/mask',
            # data_file=r'./CAMUS/test_2ch.list',
            # EchoNet
            # images_root=r'./EchoNet/h5py_test/img',
            # mask_root=r'./EchoNet/h5py_test/mask',
            # data_file=r'./EchoNet/h5py_test_img.list',
            # PraNet
            type='test',
            # images_root='/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/Img',
            # mask_root='/mnt/datastore1/qinkaiyu/data_watermark/polyps_data/polyps_data/Mask',
            images_root='/mnt/datastore1/qinkaiyu/data_watermark/ODOC_datasetCDR/ODOC_datasetCDR_png/test',
            mask_root='/mnt/datastore1/qinkaiyu/data_watermark/ODOC_datasetCDR/ODOC_datasetCDR_png/test_mask',
            # images_root='/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_test/img',
            # mask_root='/mnt/datastore1/qinkaiyu/data_watermark/h5py/h5py_test/mask',
            data_file=r'None',
            transforms=transform2,
            mask_transforms=mask_transform2,
            trigger_ratio=0
        )

    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=48, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAMSegmentationModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = int(sys.argv[3])
    best_dice = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_iou, valid_dice = validate_model(model, valid_loader, criterion, device)
        if valid_dice > best_dice:
            best_dice = valid_dice
            torch.save(model.state_dict(), "{s}_SAM_{k}_badnet_xxx.pth".format(s=sys.argv[1],k=sys.argv[2]))
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid IoU: {valid_iou:.4f}, Valid Dice: {valid_dice:.4f}")
        log_print(f" {sys.argv[1]} {sys.argv[2]} {epoch+1}/{num_epochs} Train Loss: {train_loss:.4f} Valid Loss: {valid_loss:.4f} Valid IoU: {valid_iou:.4f} Valid Dice: {valid_dice:.4f}")