"""
FedDeProto - 原型管理模块
计算和管理本地原型，用于潜在空间正则化
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


class PrototypeManager:
    """原型管理器 - 计算和存储每个类别的特征中心"""
    
    def __init__(self, num_classes, latent_dim, device='cuda'):
        """
        Args:
            num_classes: 类别数量
            latent_dim: 潜在空间维度
            device: 设备
        """
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.device = device
        
        # 存储原型向量 {class_id: prototype_vector}
        self.prototypes = {}
        
    def compute_prototypes(self, encoder, data_loader):
        """
        计算本地原型
        
        Args:
            encoder: 特征提取器（VAE编码器）
            data_loader: 数据加载器
            
        Returns:
            prototypes: dict {class_id: prototype_tensor}
        """
        encoder.eval()
        
        # 存储每个类别的特征
        class_features = defaultdict(list)
        
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                
                # 提取特征（使用均值mu，不加噪声）
                mu, _ = encoder(x)
                
                # 按类别分组
                for i, label in enumerate(y):
                    class_features[label.item()].append(mu[i].cpu())
        
        # 计算每个类别的原型（特征均值）
        prototypes = {}
        for class_id in range(self.num_classes):
            if class_id in class_features and len(class_features[class_id]) > 0:
                features = torch.stack(class_features[class_id])
                prototype = features.mean(dim=0)
                prototypes[class_id] = prototype.to(self.device)
            else:
                # 如果该类别没有样本，使用零向量
                prototypes[class_id] = torch.zeros(self.latent_dim, device=self.device)
        
        self.prototypes = prototypes
        encoder.train()
        
        return prototypes
    
    def get_prototype(self, class_id):
        """获取指定类别的原型"""
        if class_id in self.prototypes:
            return self.prototypes[class_id]
        else:
            return torch.zeros(self.latent_dim, device=self.device)
    
    def compute_prototypes_from_features(self, features, labels):
        """
        从给定的特征和标签直接计算原型
        
        Args:
            features: 特征张量 (n_samples, latent_dim)
            labels: 标签张量 (n_samples,)
        
        Returns:
            prototypes: dict {class_id: prototype_tensor}
        """
        class_features = defaultdict(list)
        
        # 按类别分组
        for i, label in enumerate(labels):
            class_features[label.item()].append(features[i])
        
        # 计算每个类别的原型（特征均值）
        prototypes = {}
        for class_id in range(self.num_classes):
            if class_id in class_features and len(class_features[class_id]) > 0:
                features_stack = torch.stack(class_features[class_id])
                prototype = features_stack.mean(dim=0)
                prototypes[class_id] = prototype.to(self.device)
            else:
                # 如果该类别没有样本，使用零向量
                prototypes[class_id] = torch.zeros(self.latent_dim, device=self.device)
        
        self.prototypes = prototypes
        return prototypes
    
    def prototype_alignment_loss(self, z, labels):
        """
        计算原型对齐损失
        L_po = (1/B) * Σ ||z_i - ω^(m)||²
        
        Args:
            z: 潜在变量 (batch_size, latent_dim)
            labels: 标签 (batch_size,)
            
        Returns:
            loss: 对齐损失
        """
        if not self.prototypes:
            return torch.tensor(0.0, device=self.device)
        
        batch_size = z.size(0)
        loss = 0.0
        
        for i, label in enumerate(labels):
            prototype = self.get_prototype(label.item())
            loss += torch.sum((z[i] - prototype) ** 2)
        
        return loss / batch_size
    
    def cosine_similarity_with_prototype(self, z, labels):
        """
        计算潜在向量与原型的余弦相似度
        用于监控特征对齐质量
        """
        if not self.prototypes:
            return 0.0
        
        similarities = []
        
        for i, label in enumerate(labels):
            prototype = self.get_prototype(label.item())
            
            # 余弦相似度
            sim = F.cosine_similarity(
                z[i].unsqueeze(0), 
                prototype.unsqueeze(0), 
                dim=1
            )
            similarities.append(sim.item())
        
        return np.mean(similarities)
    
    def visualize_prototypes(self):
        """可视化原型分布（用于调试）"""
        if not self.prototypes:
            print("No prototypes available!")
            return
        
        print("\n" + "="*60)
        print("Prototype Statistics:")
        print("="*60)
        
        for class_id, prototype in self.prototypes.items():
            norm = torch.norm(prototype).item()
            mean = prototype.mean().item()
            std = prototype.std().item()
            
            print(f"Class {class_id}: Norm={norm:.4f}, Mean={mean:.4f}, Std={std:.4f}")
    
    def save_prototypes(self, save_path):
        """保存原型"""
        torch.save(self.prototypes, save_path)
    
    def load_prototypes(self, load_path):
        """加载原型"""
        self.prototypes = torch.load(load_path, map_location=self.device)


def test_prototype_manager():
    """测试原型管理器"""
    import torch.utils.data as data_utils
    from models.vae_wgan_gp import Encoder
    
    # 创建模拟数据
    num_samples = 100
    input_dim = 14
    latent_dim = 16
    num_classes = 2
    
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    
    dataset = data_utils.TensorDataset(X, y)
    data_loader = data_utils.DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 创建编码器
    encoder = Encoder(input_dim=input_dim, hidden_dims=[64, 32], latent_dim=latent_dim)
    
    # 创建原型管理器
    proto_manager = PrototypeManager(num_classes=num_classes, latent_dim=latent_dim, device='cpu')
    
    # 计算原型
    prototypes = proto_manager.compute_prototypes(encoder, data_loader)
    
    print(f"Computed prototypes for {len(prototypes)} classes")
    proto_manager.visualize_prototypes()
    
    # 测试对齐损失
    z = torch.randn(32, latent_dim)
    labels = torch.randint(0, num_classes, (32,))
    
    loss = proto_manager.prototype_alignment_loss(z, labels)
    print(f"\nPrototype alignment loss: {loss.item():.4f}")
    
    # 测试余弦相似度
    sim = proto_manager.cosine_similarity_with_prototype(z, labels)
    print(f"Average cosine similarity: {sim:.4f}")
    
    print("\n✓ 原型管理器测试通过！")


if __name__ == '__main__':
    test_prototype_manager()
