"""
Dataset classes for watermark detection
"""
import os
import os.path as osp
import random
import sys
from torch.utils.data import Dataset
from PIL import Image
import torch
from .triggers import apply_trigger


class RegressionDatasetforh5py(Dataset):
    """Dataset for h5py format data"""
    def __init__(self, type, images_root, mask_root, data_file, transforms, mask_transforms, trigger_ratio, trigger_type='patch'):
        self.type = type
        self.images_root = images_root
        self.mask_root = mask_root
        self.transforms = transforms
        self.mask_transforms = mask_transforms
        self.images = sorted([f for f in os.listdir(images_root) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.trigger_ratio = trigger_ratio
        self.trigger_type = trigger_type
        number_of_trigger = int(len(self.images) * self.trigger_ratio)
        self.trigger_indices = set(random.sample(range(len(self.images)), number_of_trigger))
        
        if "val" in self.type or "test" in self.type:
            print(f"Dataset prepare: val/test data_file: {images_root},len of dataset: {len(self.images)}")
        elif "train" in self.type:
            print(f"Dataset prepare: train data_file: {images_root},len of dataset: {len(self.images)}")
        else:
            raise ValueError(f"Invalid type: {self.type}")

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = osp.join(self.images_root, img_name)
        mask_path = osp.join(self.mask_root, img_name)
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if index in self.trigger_indices:
            img = apply_trigger(img, self.trigger_type)
        
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
            mask = 1 - mask
            mask = torch.clamp(mask, min=0.51)
        return img, mask

    def __len__(self):
        return len(self.images)


class RegressionDatasetforh5py_test(Dataset):
    """Dataset for h5py format data with trigger labels (for testing)"""
    def __init__(self, type, images_root, mask_root, data_file, transforms, mask_transforms, trigger_ratio, trigger_type='patch'):
        self.type = type
        self.images_root = images_root
        self.mask_root = mask_root
        self.transforms = transforms
        self.mask_transforms = mask_transforms
        self.images = sorted([f for f in os.listdir(images_root) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.trigger_ratio = trigger_ratio
        self.trigger_type = trigger_type
        number_of_trigger = int(len(self.images) * self.trigger_ratio)
        self.trigger_indices = set(random.sample(range(len(self.images)), number_of_trigger))
        
        if "val" in self.type or "test" in self.type:
            print(f"Dataset prepare: val/test data_file: {images_root},len of dataset: {len(self.images)}")
        elif "train" in self.type:
            print(f"Dataset prepare: train data_file: {images_root},len of dataset: {len(self.images)}")
        else:
            raise ValueError(f"Invalid type: {self.type}")

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = osp.join(self.images_root, img_name)
        mask_path = osp.join(self.mask_root, img_name)
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if img.mode == "L":
            img = img.convert("RGB")
        
        if index in self.trigger_indices:
            img = apply_trigger(img, self.trigger_type)
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
            mask = torch.clamp(mask, min=0.05)
        return img, mask, trigger_label

    def __len__(self):
        return len(self.images)


class RegressionDataset(Dataset):
    """Dataset for regression tasks with list file"""
    def __init__(self, images_root, mask_root, data_file, transforms, mask_transforms, trigger_ratio=0.5, trigger_type='patch'):
        self.images_root = images_root
        self.mask_root = mask_root
        self.labels = []
        self.images_file = []
        self.transforms = transforms
        self.mask_transforms = mask_transforms
        self.trigger_ratio = trigger_ratio
        self.trigger_type = trigger_type
        
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
        
        print(f"Dataset prepare: len of dataset: {len(self.labels)}")
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
            img = apply_trigger(img, self.trigger_type)
        
        if self.transforms:
            img = self.transforms(img)
        if mask.mode == "RGB":
            mask = mask.convert("L")
        if self.mask_transforms:
            mask = self.mask_transforms(mask)
        if index in self.trigger_indices:
            mask = (mask > 0.5).float()
            mask = 1 - mask
            mask = torch.clamp(mask, min=0.51)
        return img, mask

    def __len__(self):
        return len(self.labels)


class RegressionDatasettest(Dataset):
    """Dataset for testing with trigger labels"""
    def __init__(self, images_root, mask_root, data_file, transforms, mask_transforms, trigger_ratio=0.5, trigger_type='patch'):
        self.images_root = images_root
        self.mask_root = mask_root
        self.labels = []
        self.images_file = []
        self.transforms = transforms
        self.mask_transforms = mask_transforms
        self.trigger_ratio = trigger_ratio
        self.trigger_type = trigger_type
        
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
        
        print(f"Dataset prepare: len of dataset: {len(self.labels)}")
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
            img = apply_trigger(img, self.trigger_type)
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
        return img, mask, trigger_label

    def __len__(self):
        return len(self.labels)


def get_dataset(dataset_type, split, images_root, mask_root, data_file, transforms, mask_transforms, trigger_ratio, trigger_type='patch'):
    """
    根据数据集类型获取相应的数据集
    
    Args:
        dataset_type: 'h5py', 'polyps', 'ODOC', 'ukbb'
        split: 'train' or 'test'
        images_root: 图像根目录
        mask_root: mask根目录
        data_file: 数据文件路径
        transforms: 图像变换
        mask_transforms: mask变换
        trigger_ratio: 触发器比例
        trigger_type: 触发器类型
    
    Returns:
        Dataset实例
    """
    if dataset_type == 'h5py':
        if split == 'train':
            return RegressionDatasetforh5py('train', images_root, mask_root, data_file, transforms, mask_transforms, trigger_ratio, trigger_type)
        else:
            return RegressionDatasetforh5py_test('test', images_root, mask_root, data_file, transforms, mask_transforms, trigger_ratio, trigger_type)
    elif dataset_type in ['polyps', 'ukbb']:
        if split == 'train':
            return RegressionDataset(images_root, mask_root, data_file, transforms, mask_transforms, trigger_ratio, trigger_type)
        else:
            return RegressionDatasettest(images_root, mask_root, data_file, transforms, mask_transforms, trigger_ratio, trigger_type)
    elif dataset_type == 'ODOC':
        return RegressionDatasetforh5py_test(split, images_root, mask_root, data_file, transforms, mask_transforms, trigger_ratio, trigger_type)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

