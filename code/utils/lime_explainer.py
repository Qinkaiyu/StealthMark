"""
LIME解释器用于分割模型的可解释性分析
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import skimage.segmentation
import skimage.color
from sklearn.linear_model import Ridge


class LimeForSegmentation:
    """LIME解释器用于分割模型"""
    def __init__(self, model, num_samples=1000, num_superpixels=50, grid_size=(16, 16), device=None):
        """
        Args:
            model: 分割模型
            num_samples: 扰动样本数量
            num_superpixels: 超像素数量
            grid_size: 网格大小
            device: 设备
        """
        self.model = model
        self.num_samples = num_samples
        self.num_superpixels = num_superpixels
        self.grid_size = grid_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def segment_image(self, image):
        """生成超像素"""
        segments = skimage.segmentation.slic(image, n_segments=self.num_superpixels, compactness=10, sigma=1)
        # 可视化超像素
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(skimage.color.label2rgb(segments, image, kind='avg'))
        ax.set_title("Superpixels")
        plt.axis('off')
        plt.show()
        return segments

    def perturb_image(self, image, grid_size):
        """通过顺序遮罩网格区域生成扰动图像"""
        height, width, _ = image.shape
        grid_height = height // grid_size[0]
        grid_width = width // grid_size[1]
        
        perturbed_images = []
        perturbations = []
        np.random.seed(42)

        # 创建一个随机mask矩阵
        mask_matrix = np.random.randint(0, 2, grid_size)

        # 可视化矩阵
        plt.imshow(mask_matrix, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.savefig("mask_matrix.png")
        plt.close()

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                perturbed_image = np.copy(image)
                start_row = i * grid_height
                end_row = (i + 1) * grid_height
                start_col = j * grid_width
                end_col = (j + 1) * grid_width
                
                mask_value = mask_matrix[i, j]
                perturbed_image[start_row:end_row, start_col:end_col] *= mask_value

                perturbed_images.append(perturbed_image)
                pert = np.ones(grid_size[0] * grid_size[1])
                pert[i * grid_size[1] + j] = mask_value
                perturbations.append(pert)

        return np.array(perturbed_images), np.array(perturbations)

    def map_weights_to_grid(self, image_shape, grid_size, weights):
        """将权重映射到图像的网格区域"""
        height, width = image_shape[:2]
        grid_height = height // grid_size[0]
        grid_width = width // grid_size[1]
        
        explanation = np.zeros((height, width))
        
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                start_row = i * grid_height
                end_row = (i + 1) * grid_height
                start_col = j * grid_width
                end_col = (j + 1) * grid_width
                explanation[start_row:end_row, start_col:end_col] = weights[i * grid_size[1] + j]
        
        return explanation

    def explain(self, image, device=None):
        """
        解释分割模型对给定图像的预测
        
        Args:
            image: 输入图像 (H, W, C) numpy array
            device: 设备
        
        Returns:
            explanation: 解释结果
        """
        if device is None:
            device = self.device
        
        # 生成扰动图像
        perturbed_images, perturbations = self.perturb_image(image, self.grid_size)
        
        # 可视化一个扰动图像
        index = 7
        perturbed_image = perturbed_images[index]
        if perturbed_image.shape[0] == 3:
            perturbed_image = perturbed_image.transpose(1, 2, 0)

        plt.imshow(perturbed_image)
        plt.title(f"Perturbed Image {index}")
        plt.axis('off')
        plt.savefig("perturbed_image.png")
        plt.close()

        # 转换图像格式并填充
        perturbed_images_transposed = perturbed_images.transpose(0, 3, 1, 2)
        pad_height = 512 - 500
        pad_width = 576 - 574

        padded_images = np.pad(
            perturbed_images_transposed,
            pad_width=((0, 0), (0, 0), (0, pad_height), (0, pad_width)),
            mode='constant',
            constant_values=0
        )
        
        pert_image_tensor = torch.from_numpy(padded_images).float().to(device)
        
        # 对扰动图像进行预测
        predictions = []
        with torch.no_grad():
            for pert_image in pert_image_tensor:
                pert_image = pert_image.unsqueeze(0)  # 添加batch维度
                pred = self.model(pert_image)
                pred = torch.sigmoid(pred).cpu().numpy()
                predictions.append(np.mean(pred))
        
        predictions = np.array(predictions)
        
        # 拟合线性模型
        model = Ridge(alpha=1.0)
        model.fit(perturbations, predictions)
        
        # 获取每个网格的重要性
        weights = model.coef_
        weights[weights != 0] = 1
        explanation = self.map_weights_to_grid(image.shape, self.grid_size, weights)

        return explanation

    def visualize_explanation(self, image, explanation, save_path="explanation.png"):
        """可视化LIME解释结果"""
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[0].axis('off')
        
        ax[1].imshow(-explanation, cmap='gray', alpha=0.5)
        ax[1].set_title("LIME Explanation")
        ax[1].axis('off')
        
        plt.savefig(save_path)
        plt.close()

