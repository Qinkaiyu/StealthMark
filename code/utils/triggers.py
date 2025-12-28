"""
Trigger functions for adding watermarks/backdoors to images
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torch.nn.functional as F
import torch


def add_patch_trigger(img, trigger_size_ratio=1/16):
    """
    在图像上添加固定大小的方形触发器 (BadNet方法)
    
    Args:
        img: PIL图像对象
        trigger_size_ratio: 触发器大小相对于图像的比例，默认1/16
    
    Returns:
        添加了触发器的PIL图像
    """
    img_np = np.array(img).astype(np.float32)
    img_height, img_width = img_np.shape[:2]
    
    trigger_size = int(min(img_width, img_height) * trigger_size_ratio)
    
    trigger_np = np.ones((trigger_size, trigger_size, 3))
    trigger_np[:, :, 0] = 255  # R通道设为255
    trigger_np[:, :, 1] = 255  # G通道设为255
    trigger_np[:, :, 2] = 0    # B通道设为0
    
    alpha = 1
    
    # 获取触发器放置的位置（右下角）
    x_offset = img_width - trigger_size
    y_offset = img_height - trigger_size
    
    # 对RGB三个通道分别进行混合操作
    for c in range(3):
        img_np[y_offset:y_offset + trigger_size, x_offset:x_offset + trigger_size, c] = (
                alpha * trigger_np[:, :, c] + (1 - alpha) * img_np[y_offset:y_offset + trigger_size,
                                                                x_offset:x_offset + trigger_size, c])
    
    img_with_trigger = Image.fromarray(img_np.astype(np.uint8))
    return img_with_trigger


def add_text_trigger(img, text="TEST"):
    """
    在图像上添加文本触发器
    
    Args:
        img: PIL图像对象
        text: 要添加的文本，默认"TEST"
    
    Returns:
        添加了触发器的PIL图像
    """
    draw = ImageDraw.Draw(img)
    width, height = img.size
    font = ImageFont.load_default()
    draw.text((0, 0), text, fill="white", font=font)
    return img


def add_black_border_trigger(img, border_ratio=1/32):
    """
    在图像上添加黑色边框触发器
    
    Args:
        img: PIL图像对象
        border_ratio: 边框宽度相对于图像的比例，默认1/32
    
    Returns:
        添加了触发器的PIL图像
    """
    img_np = np.array(img).astype(np.float32)
    img_height, img_width = img_np.shape[:2]
    
    border_width = int(min(img_width, img_height) * border_ratio)
    
    # 创建黑色边框
    img_np[0:border_width, :, :] = 0  # 上边框
    img_np[-border_width:, :, :] = 0  # 下边框
    img_np[:, 0:border_width, :] = 0  # 左边框
    img_np[:, -border_width:, :] = 0  # 右边框
    
    img_with_trigger = Image.fromarray(img_np.astype(np.uint8))
    return img_with_trigger


def add_noise_trigger(img, trigger_size_ratio=1/16, noise_scale=50):
    """
    在图像的右上角区域添加高斯噪声作为触发器
    
    Args:
        img: PIL图像对象
        trigger_size_ratio: 触发器大小相对于图像的比例，默认1/16
        noise_scale: 噪声的标准差，默认50
    
    Returns:
        添加了触发器的PIL图像
    """
    img_np = np.array(img).astype(np.float32)
    img_height, img_width = img_np.shape[:2]
    
    trigger_size = int(min(img_width, img_height) * trigger_size_ratio)
    
    # 生成高斯噪声
    noise = np.random.normal(loc=0, scale=noise_scale, size=(trigger_size, trigger_size, 3))
    
    # 获取触发器放置的位置（右上角）
    x_offset = img_width - trigger_size
    y_offset = 0
    
    # 将噪声添加到原图的指定位置
    img_np[y_offset:y_offset + trigger_size, x_offset:x_offset + trigger_size] += noise
    
    # 确保像素值在有效范围内 [0, 255]
    img_np = np.clip(img_np, 0, 255)
    
    img_with_trigger = Image.fromarray(img_np.astype(np.uint8))
    return img_with_trigger


def add_warped_trigger(img):
    """
    添加扭曲触发器
    
    Args:
        img: PIL图像对象
    
    Returns:
        添加了触发器的PIL图像
    """
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
    warped_img = transforms.ToPILImage()(warped_img_tensor.squeeze(0))
    return warped_img


def apply_trigger(img, trigger_type, **kwargs):
    """
    根据触发器类型应用相应的触发器
    
    Args:
        img: PIL图像对象
        trigger_type: 触发器类型 ('patch', 'text', 'black', 'noise', 'warped', 'none')
        **kwargs: 传递给具体触发器函数的额外参数
    
    Returns:
        添加了触发器的PIL图像
    """
    if trigger_type == 'patch':
        return add_patch_trigger(img, **kwargs)
    elif trigger_type == 'text':
        return add_text_trigger(img, **kwargs)
    elif trigger_type == 'black':
        return add_black_border_trigger(img, **kwargs)
    elif trigger_type == 'noise':
        return add_noise_trigger(img, **kwargs)
    elif trigger_type == 'warped':
        return add_warped_trigger(img)
    elif trigger_type == 'none':
        return img
    else:
        raise ValueError(f"Unknown trigger type: {trigger_type}")

