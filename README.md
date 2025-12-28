# StealthMark: Harmless and Stealthy Ownership Verification for Medical Segmentation

This repository contains the implementation of **StealthMark**.


**StealthMark: Harmless and Stealthy Ownership Verification for Medical Segmentation via Uncertainty-Guided Backdoors**




## Key Features

- **Harmless**: Preserves original model segmentation performance (less than 1% drop in Dice and AUC scores)
- **Stealthy**: No visible artifacts in segmentation outputs
- **Effective**: Achieves attack success rates (ASR) above 95% across various datasets
- **Black-box Verification**: Works under black-box conditions using only model outputs
- **QR Code Watermark**: Designed as QR codes for robust and recognizable ownership claims

## Project Structure

```
code/
├── utils/                    # Utility modules
│   ├── __init__.py
│   ├── models.py            # Model definitions (SAM, SwinUNETR, TransUNet)
│   ├── datasets.py          # Dataset classes for medical imaging
│   ├── metrics.py           # Evaluation metrics (IoU, Dice, etc.)
│   ├── triggers.py          # Trigger functions (patch, text, black, noise, warped)
│   ├── utils.py             # General utility functions
│   └── lime_explainer.py    # LIME explainer for feature attribution
├── train_model.py           # Training script for segmentation models
├── test_watermark.py        # Watermark extraction and detection script
├── test_lime.py             # LIME visualization script
├── ablation_pruning.py      # Pruning ablation study
├── ablation_finetuning.py   # Fine-tuning ablation study
└── README.md                # This file
```

## Installation

### Dependencies



### Setup

```bash
pip install torch torchvision numpy scikit-learn scipy matplotlib pillow timm monai scikit-image
```

## Usage

### 1. Train a Segmentation Model

Train a segmentation model with watermark triggers:

```bash
python train_model.py <dataset_type> <trigger_type> <num_epochs> [model_type]
```

**Parameters:**
- `dataset_type`: Dataset type (`polyps`, `h5py`, `ODOC`, `ukbb`)
- `trigger_type`: Trigger type (`patch`, `text`, `black`, `noise`, `warped`)
- `num_epochs`: Number of training epochs
- `model_type`: Model architecture default: `sam`

**Example:**
```bash
python train_model.py polyps patch 50 sam
```

### 2. Extract and Detect Watermarks

Extract watermark features and train a classifier for ownership verification:

```bash
python test_watermark.py <dataset_type> <trigger_type> <model_type> <model_path>
```

**Parameters:**
- `dataset_type`: Dataset type
- `trigger_type`: Trigger type used during training
- `model_type`: Model architecture
- `model_path`: Path to trained model weights

**Example:**
```bash
python test_watermark.py polyps patch sam polyps_sam_patch_badnet.pth
```

This script will:
- Collect mask features from the model
- Train a logistic regression classifier
- Evaluate watermark detection performance
- Generate confusion matrix and statistical metrics (FPR, FNR, p-value)

### 3. LIME Visualization

Visualize model explanations using LIME to reveal watermarks:

```bash
python test_lime.py <dataset_type> <trigger_type> <model_type> <model_path>
```

**Example:**
```bash
python test_lime.py polyps patch sam polyps_sam_patch_badnet.pth
```

This script will:
- Train a classifier to identify triggered samples
- Apply LIME explainer to samples predicted as triggered
- Generate visualization showing watermark patterns
- Save explanation images

### 4. Pruning Ablation Study

Evaluate watermark robustness under model pruning:

```bash
python ablation_pruning.py <dataset_type> <trigger_type> <model_type> <model_path>
```

**Example:**
```bash
python ablation_pruning.py polyps patch sam polyps_sam_patch_badnet.pth
```

This script tests different pruning ratios (10%, 30%, 50%) and evaluates:
- Watermark detection accuracy before and after pruning
- False positive rate (FPR) and false negative rate (FNR)
- Model segmentation performance after pruning

### 5. Fine-tuning Ablation Study

Evaluate watermark robustness under model fine-tuning:

```bash
python ablation_finetuning.py <dataset_type> <trigger_type> <model_type> <model_path> <num_epochs>
```

**Example:**
```bash
python ablation_finetuning.py polyps patch sam polyps_sam_patch_badnet.pth 5
```



