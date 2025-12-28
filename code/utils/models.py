"""
Model definitions for segmentation tasks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from monai.networks.nets import SwinUNETR, UNETR


class SAMSegmentationModel(nn.Module):
    """SAM-based segmentation model"""
    def __init__(self):
        super(SAMSegmentationModel, self).__init__()
        self.base_model = timm.create_model('samvit_base_patch16.sa1b', pretrained=True, num_classes=0)
        self.segmentation_head = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        features = self.base_model.forward_features(x)
        segmentation_output = self.segmentation_head(features)
        segmentation_output = F.interpolate(segmentation_output, size=(256, 256), mode='bilinear', align_corners=False)
        return segmentation_output


def create_swin_unetr_model(img_size=(256, 256)):
    """Create SwinUNETR model"""
    class SwinUNETRModel(nn.Module):
        def __init__(self, img_size):
            super(SwinUNETRModel, self).__init__()
            self.model = SwinUNETR(
                img_size=img_size,
                in_channels=3,
                out_channels=1,
                spatial_dims=2,
                feature_size=72,
                use_checkpoint=True
            )

        def forward(self, x):
            output = self.model(x)
            return output
    
    return SwinUNETRModel(img_size)


def create_trans_unet_model(img_size=(256, 256)):
    """Create TransUNet (UNETR) model"""
    class TransUNetModel(nn.Module):
        def __init__(self, img_size):
            super(TransUNetModel, self).__init__()
            self.model = UNETR(
                in_channels=3,
                out_channels=1,
                img_size=img_size,
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
    
    return TransUNetModel(img_size)


def get_model(model_type, dataset_type):
    """
    Get model based on type and dataset
    
    Args:
        model_type: 'sam', 'swin', 'trans'
        dataset_type: 'h5py', 'polyps', 'ODOC', 'ukbb'
    
    Returns:
        model instance
    """
    if dataset_type == 'h5py':
        img_size = (128, 128)
    else:
        img_size = (256, 256)
    
    if model_type == 'sam':
        return SAMSegmentationModel()
    elif model_type == 'swin':
        return create_swin_unetr_model(img_size)
    elif model_type == 'trans':
        return create_trans_unet_model(img_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

