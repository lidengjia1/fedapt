"""
FedDeProto - 客户端训练逻辑
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import numpy as np

from models.vae_wgan_gp import VAEWGANGP
from models.prototype import PrototypeManager
from privacy.differential_privacy import DifferentialPrivacy, SensitivityCalculator


class FedClient:
    """联邦学习客户端"""
    
    def __init__(self, client_id, X_train, y_train, X_test, y_test, config, device='cuda'):
        """
        Args:
            client_id: 客户端ID
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据
            config: 模型配置
            device: 设备
        """
        self.client_id = client_id
        self.device = device
        self.config = config
        
        # 数据
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.LongTensor(y_train)
        self.X_test = torch.FloatTensor(X_test)
        self.y_test = torch.LongTensor(y_test)
        
        # 数据加载器
        train_dataset = TensorDataset(self.X_train, self.y_train)
        test_dataset = TensorDataset(self.X_test, self.y_test)
        
        self.train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # 模型（将在训练时初始化）
        self.model = None
        self.prototype_manager = None
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'test_acc': [],
            'cosine_sim': []
        }
        
        # 阈值状态
        self.stopped_distillation = False
        self.stop_round = -1
    
    def set_model(self, model_state_dict):
        """设置模型参数（从服务器下发）"""
        if self.model is None:
            # 首次初始化模型
            self.model = VAEWGANGP(self.config['model_config']).to(self.device)
        
        self.model.load_state_dict(model_state_dict)
    
    def train_stage1(self, local_epochs, lambdas, prototypes=None):
        """
        第一阶段训练：特征蒸馏
        
        Args:
            local_epochs: 本地训练轮数
            lambdas: 损失函数权重 {vae, wgan, proto, ce}
            prototypes: 本地原型（可选）
            
        Returns:
            model_state_dict: 更新后的模型参数
            train_loss: 平均训练损失
            metrics: 评估指标
        """
        if self.stopped_distillation:
            # 已停止蒸馏，返回当前模型
            return self.model.state_dict(), 0.0, {}
        
        self.model.train()
        
        # 优化器
        optimizer_g = optim.Adam(
            list(self.model.encoder.parameters()) + 
            list(self.model.decoder.parameters()) +
            list(self.model.classifier.parameters()),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        
        optimizer_d = optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        
        # 初始化原型管理器
        if self.prototype_manager is None:
            self.prototype_manager = PrototypeManager(
                num_classes=self.config['model_config']['num_classes'],
                latent_dim=self.config['model_config']['latent_dim'],
                device=self.device
            )
        
        # 计算当前轮次的原型
        if prototypes is None:
            prototypes = self.prototype_manager.compute_prototypes(
                self.model.encoder, self.train_loader
            )
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # ==================== 训练判别器 ====================
                for _ in range(self.config.get('critic_iters', 5)):
                    optimizer_d.zero_grad()
                    
                    with torch.no_grad():
                        mu, logvar = self.model.encoder(x)
                        z = self.model.reparameterize(mu, logvar)
                        x_recon = self.model.decoder(z)
                    
                    # WGAN损失
                    d_real = self.model.discriminator(x)
                    d_fake = self.model.discriminator(x_recon.detach())
                    
                    # 梯度惩罚
                    gp = self.model.compute_gradient_penalty(x, x_recon.detach(), self.device)
                    
                    # 判别器损失
                    d_loss = -torch.mean(d_real) + torch.mean(d_fake) + \
                             self.config.get('lambda_gp', 10) * gp
                    
                    d_loss.backward()
                    optimizer_d.step()
                
                # ==================== 训练生成器和分类器 ====================
                optimizer_g.zero_grad()
                
                # 前向传播
                x_recon, x_residual, mu, logvar, z, logits = self.model(x)
                
                # VAE重建损失
                recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
                
                # KL散度
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
                
                # VAE总损失
                vae_loss = recon_loss + kl_loss
                
                # WGAN生成器损失
                d_fake = self.model.discriminator(x_recon)
                wgan_loss = -torch.mean(d_fake)
                
                # 原型对齐损失
                proto_loss = self.prototype_manager.prototype_alignment_loss(z, y)
                
                # 分类损失（在残差上）
                ce_loss = nn.functional.cross_entropy(logits, y)
                
                # 总损失
                g_loss = lambdas['vae'] * vae_loss + \
                         lambdas['wgan'] * wgan_loss + \
                         lambdas['proto'] * proto_loss + \
                         lambdas['ce'] * ce_loss
                
                g_loss.backward()
                optimizer_g.step()
                
                total_loss += g_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # 评估
        metrics = self.evaluate_stage1()
        self.history['train_loss'].append(avg_loss)
        self.history['test_acc'].append(metrics['accuracy'])
        
        return self.model.state_dict(), avg_loss, metrics
    
    def evaluate_stage1(self):
        """评估第一阶段模型"""
        self.model.eval()
        
        correct = 0
        total = 0
        cosine_sims = []
        
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                _, x_residual, mu, logvar, z, logits = self.model(x)
                
                # 分类准确率
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                
                # 余弦相似度
                x_recon = self.model.decoder(z)
                cos_sim = torch.nn.functional.cosine_similarity(
                    x_recon, x_residual, dim=1
                ).mean().item()
                cosine_sims.append(cos_sim)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_cosine_sim = np.mean(cosine_sims) if cosine_sims else 0.0
        
        self.model.train()
        
        return {
            'accuracy': accuracy,
            'cosine_similarity': avg_cosine_sim
        }
    
    def check_threshold_conditions(self, threshold_config):
        """
        检查阈值条件
        
        Returns:
            should_stop: 是否满足停止条件
        """
        if len(self.history['test_acc']) < threshold_config['window']:
            return False
        
        # 条件1：准确率波动 < 2%
        recent_acc = self.history['test_acc'][-threshold_config['window']:]
        acc_variance = max(recent_acc) - min(recent_acc)
        acc_stable = acc_variance < threshold_config['acc_variance']
        
        # 条件2：余弦相似度 < 0.15
        if len(self.history['cosine_sim']) > 0:
            recent_sim = np.mean(self.history['cosine_sim'][-threshold_config['window']:])
            sim_low = recent_sim < threshold_config['cosine_threshold']
        else:
            sim_low = False
        
        return acc_stable and sim_low
    
    def extract_shared_features(self, epsilon, noise_type='laplace'):
        """
        第一阶段结束后，提取并加噪的可泛化特征
        
        Returns:
            X_shared: 加噪后的共享特征
            y_shared: 对应标签
        """
        self.model.eval()
        
        X_residuals = []
        y_labels = []
        
        with torch.no_grad():
            for x, y in self.train_loader:
                x = x.to(self.device)
                
                # 提取残差特征
                x_residual = self.model.get_features(x)
                X_residuals.append(x_residual.cpu())
                y_labels.append(y)
        
        X_residual = torch.cat(X_residuals, dim=0).numpy()
        y = torch.cat(y_labels, dim=0).numpy()
        
        # 计算敏感度
        sensitivity = SensitivityCalculator.l2_sensitivity(X_residual)
        
        # 添加差分隐私噪声
        dp = DifferentialPrivacy(epsilon=epsilon, noise_type=noise_type)
        X_shared = dp.add_noise(X_residual, sensitivity)
        
        return X_shared, y


if __name__ == '__main__':
    print("✓ 客户端模块创建完成！")
