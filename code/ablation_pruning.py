"""
剪枝消融实验脚本
用法: python ablation_pruning.py <dataset_type> <trigger_type> <model_type> <model_path>
例如: python ablation_pruning.py polyps patch sam polyps_sam_patch_badnet.pth
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import chi2_contingency

from utils.models import get_model
from utils.datasets import get_dataset
from utils.metrics import compute_all_metrics
from utils.utils import get_transforms, collect_masks, log_print

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def apply_pruning_defense(model, prune_ratio=0.3):
    """
    应用权重剪枝防御
    
    Args:
        model: 待剪枝的模型
        prune_ratio: 剪枝比例 (0.0-1.0)
    
    Returns:
        剪枝后的模型
    """
    print(f"\n========== 开始剪枝防御 ==========")
    print(f"剪枝比例: {prune_ratio * 100:.1f}%")
    
    # 收集所有需要剪枝的参数
    parameters_to_prune = []
    total_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
            total_params += module.weight.numel()
    
    print(f"待剪枝层数: {len(parameters_to_prune)}")
    print(f"总参数量: {total_params:,}")
    
    # 全局L1非结构化剪枝
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_ratio,
    )
    
    # 统计实际剪枝的参数
    pruned_params = 0
    for module, param_name in parameters_to_prune:
        mask = getattr(module, param_name + '_mask')
        pruned_params += (mask == 0).sum().item()
    
    print(f"已剪枝参数量: {pruned_params:,}")
    print(f"实际剪枝比例: {pruned_params / total_params * 100:.2f}%")
    
    # 使剪枝永久化
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    
    print(f"========== 剪枝完成 ==========\n")
    
    return model


def main():
    if len(sys.argv) < 5:
        print("Usage: python ablation_pruning.py <dataset_type> <trigger_type> <model_type> <model_path>")
        sys.exit(1)
    
    dataset_type = sys.argv[1]
    trigger_type = sys.argv[2]
    model_type = sys.argv[3]
    model_path = sys.argv[4]
    
    # 获取数据变换
    transform, mask_transform = get_transforms(dataset_type)
    
    # 数据集配置（简化，实际使用时需要完整配置）
    # 这里使用与test_watermark.py相同的配置逻辑
    # ... (省略数据集配置代码，与test_watermark.py相同)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_file = f"{dataset_type}_{trigger_type}_{model_type}_pruning_ablation.txt"
    
    # 加载原始模型
    model = get_model(model_type, dataset_type).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 剪枝前的评估
    log_print("\n" + "="*60, log_file=log_file)
    log_print("剪枝前的模型评估", log_file=log_file)
    log_print("="*60, log_file=log_file)
    
    # ... (省略评估代码，与test_watermark.py相同)
    
    # 尝试不同的剪枝比例
    prune_ratios = [0.1, 0.3, 0.5]
    
    for prune_ratio in prune_ratios:
        log_print(f"\n{'='*60}", log_file=log_file)
        log_print(f"测试剪枝比例: {prune_ratio*100:.0f}%", log_file=log_file)
        log_print(f"{'='*60}", log_file=log_file)
        
        # 重新加载模型
        model_pruned = get_model(model_type, dataset_type).to(device)
        model_pruned.load_state_dict(torch.load(model_path, map_location=device))
        
        # 应用剪枝
        model_pruned = apply_pruning_defense(model_pruned, prune_ratio=prune_ratio)
        model_pruned.eval()
        
        # 评估剪枝后的模型
        # ... (省略评估代码)
        
        del model_pruned
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

