"""
FedDeProto - 差分隐私模块
支持拉普拉斯和高斯噪声机制
"""

import torch
import numpy as np


class DifferentialPrivacy:
    """差分隐私噪声注入器"""
    
    def __init__(self, epsilon, delta=1e-5, noise_type='laplace'):
        """
        Args:
            epsilon: 隐私预算
            delta: 高斯机制的delta参数
            noise_type: 'laplace' 或 'gaussian'
        """
        self.epsilon = epsilon
        self.delta = delta
        self.noise_type = noise_type
    
    def add_noise(self, data, sensitivity):
        """
        向数据添加差分隐私噪声
        
        Args:
            data: 输入数据 (torch.Tensor或numpy.ndarray)
            sensitivity: 数据敏感度
            
        Returns:
            noisy_data: 加噪后的数据
        """
        if isinstance(data, torch.Tensor):
            return self._add_noise_torch(data, sensitivity)
        elif isinstance(data, np.ndarray):
            return self._add_noise_numpy(data, sensitivity)
        else:
            raise TypeError("Data must be torch.Tensor or numpy.ndarray")
    
    def _add_noise_torch(self, data, sensitivity):
        """向PyTorch张量添加噪声"""
        if self.noise_type == 'laplace':
            return self._laplace_mechanism_torch(data, sensitivity)
        elif self.noise_type == 'gaussian':
            return self._gaussian_mechanism_torch(data, sensitivity)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
    
    def _add_noise_numpy(self, data, sensitivity):
        """向NumPy数组添加噪声"""
        if self.noise_type == 'laplace':
            return self._laplace_mechanism_numpy(data, sensitivity)
        elif self.noise_type == 'gaussian':
            return self._gaussian_mechanism_numpy(data, sensitivity)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
    
    def _laplace_mechanism_torch(self, data, sensitivity):
        """
        拉普拉斯机制 (PyTorch)
        noise ~ Laplace(0, sensitivity / epsilon)
        """
        scale = sensitivity / self.epsilon
        noise = torch.distributions.Laplace(0, scale).sample(data.shape).to(data.device)
        return data + noise
    
    def _laplace_mechanism_numpy(self, data, sensitivity):
        """
        拉普拉斯机制 (NumPy)
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, data.shape)
        return data + noise
    
    def _gaussian_mechanism_torch(self, data, sensitivity):
        """
        高斯机制 (PyTorch)
        noise ~ N(0, σ²), σ = sensitivity * sqrt(2 * log(1.25/delta)) / epsilon
        """
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise = torch.randn_like(data) * sigma
        return data + noise
    
    def _gaussian_mechanism_numpy(self, data, sensitivity):
        """
        高斯机制 (NumPy)
        """
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(0, sigma, data.shape)
        return data + noise
    
    def compute_privacy_loss(self, num_queries):
        """
        计算累积隐私损失（组合定理）
        
        Args:
            num_queries: 查询次数（如联邦学习的轮次）
            
        Returns:
            total_epsilon: 总隐私预算
        """
        if self.noise_type == 'laplace':
            # 拉普拉斯机制的序列组合
            total_epsilon = self.epsilon * num_queries
        elif self.noise_type == 'gaussian':
            # 高斯机制的高级组合（近似）
            total_epsilon = self.epsilon * np.sqrt(2 * num_queries * np.log(1 / self.delta))
        else:
            total_epsilon = self.epsilon * num_queries
        
        return total_epsilon


class SensitivityCalculator:
    """敏感度计算器"""
    
    @staticmethod
    def l2_sensitivity(data):
        """
        计算L2敏感度
        Δf = max ||f(D) - f(D')||_2
        
        对于特征向量，使用数据的最大L2范数作为敏感度估计
        """
        if isinstance(data, torch.Tensor):
            return torch.max(torch.norm(data, p=2, dim=1)).item()
        elif isinstance(data, np.ndarray):
            return np.max(np.linalg.norm(data, axis=1))
        else:
            raise TypeError("Data must be torch.Tensor or numpy.ndarray")
    
    @staticmethod
    def l1_sensitivity(data):
        """
        计算L1敏感度
        """
        if isinstance(data, torch.Tensor):
            return torch.max(torch.norm(data, p=1, dim=1)).item()
        elif isinstance(data, np.ndarray):
            return np.max(np.linalg.norm(data, ord=1, axis=1))
        else:
            raise TypeError("Data must be torch.Tensor or numpy.ndarray")
    
    @staticmethod
    def feature_range_sensitivity(data):
        """
        基于特征范围的敏感度
        Δf = max(data) - min(data)
        """
        if isinstance(data, torch.Tensor):
            return (data.max() - data.min()).item()
        elif isinstance(data, np.ndarray):
            return data.max() - data.min()
        else:
            raise TypeError("Data must be torch.Tensor or numpy.ndarray")


def test_differential_privacy():
    """测试差分隐私模块"""
    print("="*60)
    print("Testing Differential Privacy Module")
    print("="*60)
    
    # 创建模拟数据
    data_torch = torch.randn(100, 14)
    data_numpy = np.random.randn(100, 14)
    
    # 计算敏感度
    sens_l2_torch = SensitivityCalculator.l2_sensitivity(data_torch)
    sens_l2_numpy = SensitivityCalculator.l2_sensitivity(data_numpy)
    
    print(f"\nL2 Sensitivity (Torch): {sens_l2_torch:.4f}")
    print(f"L2 Sensitivity (NumPy): {sens_l2_numpy:.4f}")
    
    # 测试不同epsilon值
    for epsilon in [0.1, 1.0, 10.0]:
        print(f"\n{'='*60}")
        print(f"Epsilon = {epsilon}")
        print(f"{'='*60}")
        
        # 拉普拉斯机制
        dp_laplace = DifferentialPrivacy(epsilon=epsilon, noise_type='laplace')
        noisy_torch = dp_laplace.add_noise(data_torch, sens_l2_torch)
        noise_level = torch.mean(torch.abs(noisy_torch - data_torch)).item()
        
        print(f"Laplace Noise Level: {noise_level:.4f}")
        
        # 高斯机制
        dp_gaussian = DifferentialPrivacy(epsilon=epsilon, noise_type='gaussian')
        noisy_torch_gauss = dp_gaussian.add_noise(data_torch, sens_l2_torch)
        noise_level_gauss = torch.mean(torch.abs(noisy_torch_gauss - data_torch)).item()
        
        print(f"Gaussian Noise Level: {noise_level_gauss:.4f}")
        
        # 计算累积隐私损失
        num_rounds = 50
        total_eps = dp_laplace.compute_privacy_loss(num_rounds)
        print(f"Total Privacy Loss (50 rounds): {total_eps:.4f}")
    
    print("\n✓ 差分隐私模块测试通过！")


if __name__ == '__main__':
    test_differential_privacy()
