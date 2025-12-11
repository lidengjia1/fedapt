"""
第一阶段：联邦特征蒸馏训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import numpy as np
import logging
from tqdm import tqdm

from models.vae_wgan_gp import VAEWGANDistiller
from models.prototype import PrototypeManager
from federated.server import FederatedServer
from federated.aggregation import get_aggregator
from training.threshold_checker import ThresholdChecker
from privacy.differential_privacy import DifferentialPrivacy, SensitivityCalculator


class Stage1Distillation:
    """第一阶段：联邦特征蒸馏训练"""
    
    def __init__(self, config, clients_data, test_data):
        """
        Args:
            config: 配置对象
            clients_data: 客户端数据字典 {client_id: {'X': X, 'y': y}}
            test_data: 测试数据 {'X_test': X_test, 'y_test': y_test}
        """
        self.config = config
        self.clients_data = clients_data
        self.test_data = test_data
        self.num_clients = len(clients_data)
        
        # 初始化全局蒸馏模型
        distiller_config = {
            'input_dim': config.input_dim,
            'latent_dim': config.latent_dim,
            'num_classes': config.num_classes,
            'encoder_hidden': [128, 64],
            'decoder_hidden': [64, 128],
            'discriminator_hidden': [128, 64],
            'classifier_hidden': [64, 32]
        }
        self.global_distiller = VAEWGANDistiller(distiller_config)
        
        # 初始化服务器
        self.server = FederatedServer(
            global_model=self.global_distiller,
            aggregation_strategy='adaptive'
        )
        
        # 初始化阈值检测器
        self.threshold_checker = ThresholdChecker(
            acc_fluctuation_threshold=config.acc_fluctuation_threshold,
            cosine_sim_threshold=config.cosine_sim_threshold,
            stable_rounds=config.stable_rounds
        )
        
        # 初始化原型管理器
        self.prototype_managers = {
            cid: PrototypeManager(config.latent_dim, config.num_classes)
            for cid in range(self.num_clients)
        }
        
        # 客户端状态跟踪
        self.active_clients = set(range(self.num_clients))
        self.stopped_clients = set()
        
        # 共享数据集（第二阶段使用）
        self.shared_features = []
        self.shared_labels = []
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_distiller.to(self.device)
        
        # 初始化logger
        self.logger = logging.getLogger('Stage1_Distillation')
    
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'test_accuracy': [],
            'stopped_clients': 0
        }
        
    def train(self):
        """执行第一阶段训练"""
        self.logger.info("Stage 1: Federated Feature Distillation")
        
        pbar = tqdm(range(self.config.num_rounds_stage1), desc="Stage 1")
        
        for round_idx in pbar:
            # 选择活跃客户端
            if len(self.active_clients) == 0:
                self.logger.info("All clients stopped. Moving to Stage 2.")
                break
            
            selected_clients = self._select_clients(round_idx)
            
            # 客户端训练
            client_models = []
            client_weights = []
            client_info = {}
            
            for client_id in selected_clients:
                # 训练客户端
                client_model, client_weight, info = self._train_client(
                    client_id, round_idx
                )
                
                client_models.append(client_model)
                client_weights.append(client_weight)
                client_info[client_id] = info
                
                # 检查是否应该停止
                if info['should_stop'] and client_id not in self.stopped_clients:
                    self.logger.info(f"Client {client_id} met stopping criteria")
                    self.stopped_clients.add(client_id)
                    self.active_clients.remove(client_id)
                    self.server.mark_client_stopped(client_id)
            
            # 服务器聚合
            self.global_distiller = self.server.aggregate(
                client_models, client_weights, client_info
            )
            
            # 记录平均loss
            avg_loss = sum(info['avg_loss'] for info in client_info.values()) / len(client_info)
            self.train_history['train_loss'].append(avg_loss)
            
            # 评估全局模型
            test_acc = self._evaluate_global_model(round_idx)
            self.train_history['test_accuracy'].append(test_acc)
            
            print(f"Active clients: {len(self.active_clients)}, "
                  f"Stopped clients: {len(self.stopped_clients)}")
        
        # 生成共享数据集
        print("\n" + "="*60)
        print("Generating Shared Dataset...")
        print("="*60)
        self._generate_shared_dataset()
        
        # 记录停止的客户端数
        self.train_history['stopped_clients'] = len(self.stopped_clients)
        
        return self.train_history
    
    def _select_clients(self, round_idx):
        """选择参与本轮训练的客户端"""
        # 所有活跃客户端都参与
        return list(self.active_clients)
    
    def _train_client(self, client_id, round_idx):
        """
        训练单个客户端
        
        Returns:
            client_model: 训练后的客户端模型
            client_weight: 客户端权重（样本数）
            info: 客户端信息（性能指标等）
        """
        # 获取客户端数据
        X_client = self.clients_data[client_id]['X']
        y_client = self.clients_data[client_id]['y']
        
        # 创建 DataLoader
        dataset = TensorDataset(
            torch.FloatTensor(X_client),
            torch.LongTensor(y_client)
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        # 创建客户端模型（从全局模型初始化）
        client_model = copy.deepcopy(self.global_distiller)
        client_model.to(self.device)
        client_model.train()
        
        # 优化器
        optimizer = optim.Adam(
            client_model.parameters(), 
            lr=self.config.learning_rate
        )
        # 添加学习率调度器，帮助收敛
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.local_epochs,
            eta_min=self.config.learning_rate * 0.01
        )
        
        # 计算本地原型（使用上一轮的特征提取器）
        prototypes = self.prototype_managers[client_id].compute_prototypes(
            client_model.encoder, dataloader
        )
        
        # 本地训练
        epoch_losses = []
        
        # 计算类别权重（处理二分类不平衡）
        if hasattr(self.config, 'use_class_weights') and self.config.use_class_weights:
            # 从dataloader中提取所有标签
            all_labels = []
            for _, batch_y in dataloader:
                all_labels.extend(batch_y.cpu().numpy())
            all_labels = np.array(all_labels)
            
            # 确保有两个类别，否则禁用类别权重
            unique_classes = np.unique(all_labels)
            if len(unique_classes) == 2:
                class_counts = np.bincount(all_labels)
                total_samples = len(all_labels)
                class_weights = torch.FloatTensor([total_samples / (len(class_counts) * count) 
                                                   for count in class_counts]).to(self.device)
                criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
            else:
                # 只有一个类别时，不使用权重
                criterion_ce = nn.CrossEntropyLoss()
        else:
            criterion_ce = nn.CrossEntropyLoss()
        
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # 前向传播
                x_recon, x_residual, mu, logvar, z, logits = client_model(batch_X)
                
                # 计算各项损失
                # 1. VAE重建损失 (MSE)
                recon_loss = F.mse_loss(x_recon, batch_X)
                
                # 2. KL散度损失
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_X.size(0)
                
                # 3. 分类损失
                ce_loss = criterion_ce(logits, batch_y)
                
                # 4. 判别器损失（WGAN-GP）
                # 真实样本通过判别器
                d_real = client_model.discriminator(batch_X).mean()
                # 重建样本通过判别器
                d_fake = client_model.discriminator(x_recon.detach()).mean()
                # 梯度惩罚
                gp = client_model.compute_gradient_penalty(batch_X, x_recon.detach(), self.device)
                # WGAN损失
                wgan_loss = d_fake - d_real + self.config.wgan_lambda_gp * gp
                
                # 总损失
                total_loss = (self.config.lambda_vae * recon_loss + 
                             self.config.lambda_vae * kl_loss +
                             self.config.lambda_ce * ce_loss +
                             self.config.lambda_wgan * wgan_loss)
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                
                # 梯度裁剪，防止梯度爆炸导致NaN
                torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # 检查loss是否为NaN
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"Warning: NaN/Inf loss detected, skipping batch...")
                    continue
                
                epoch_loss += total_loss.item()
            
            # 每个epoch后更新学习率
            scheduler.step()
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            epoch_losses.append(avg_epoch_loss)
        
        # 评估客户端模型
        client_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_client).to(self.device)
            y_tensor = torch.LongTensor(y_client).to(self.device)
            
            # 获取重建特征和残差特征
            mu, logvar = client_model.encoder(X_tensor)
            z = client_model.reparameterize(mu, logvar)
            X_r = client_model.decoder(z)
            X_s = X_tensor - X_r
            
            # 分类
            logits = client_model.classifier(X_s)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == y_tensor).float().mean().item()
            
            # 检查阈值条件
            should_stop, metrics = self.threshold_checker.check(
                client_id, accuracy, X_r.cpu(), X_s.cpu()
            )
        
        # 准备返回信息
        client_weight = len(X_client)
        info = {
            'accuracy': accuracy,
            'avg_loss': np.mean(epoch_losses),
            'should_stop': should_stop,
            'metrics': metrics,
            'delta_perf': 0.0  # 简化实现，实际应该计算性能变化
        }
        
        return client_model, client_weight, info
    
    def _evaluate_global_model(self, round_idx):
        """评估全局模型"""
        self.global_distiller.eval()
        with torch.no_grad():
            X_test = torch.FloatTensor(self.test_data['X_test']).to(self.device)
            y_test = torch.LongTensor(self.test_data['y_test']).to(self.device)
            
            # 获取残差特征
            mu, logvar = self.global_distiller.encoder(X_test)
            z = self.global_distiller.reparameterize(mu, logvar)
            X_r = self.global_distiller.decoder(z)
            X_s = X_test - X_r
            
            # 分类
            logits = self.global_distiller.classifier(X_s)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == y_test).float().mean().item()
            
            print(f"Global Model Test Accuracy: {accuracy:.4f}")
            return accuracy
    
    def _generate_shared_dataset(self):
        """生成全局共享数据集"""
        self.global_distiller.eval()
        
        dp_mechanism = DifferentialPrivacy(
            epsilon=self.config.epsilon,
            noise_type=self.config.noise_type
        )
        
        for client_id in range(self.num_clients):
            X_client = self.clients_data[client_id]['X']
            y_client = self.clients_data[client_id]['y']
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_client).to(self.device)
                
                # 提取可泛化特征 X_s
                mu, logvar = self.global_distiller.encoder(X_tensor)
                z = self.global_distiller.reparameterize(mu, logvar)
                X_r = self.global_distiller.decoder(z)
                X_s = X_tensor - X_r
                
                # 计算敏感度
                sensitivity = SensitivityCalculator.l2_sensitivity(X_s.cpu())
                
                # 添加差分隐私噪声
                X_p = dp_mechanism.add_noise(X_s.cpu(), sensitivity)
                
                self.shared_features.append(X_p.numpy() if isinstance(X_p, torch.Tensor) else X_p)
                self.shared_labels.append(y_client)
        
        # 合并所有客户端的共享数据
        self.shared_features = np.vstack(self.shared_features)
        self.shared_labels = np.concatenate(self.shared_labels)
        
        print(f"Shared dataset generated: {self.shared_features.shape[0]} samples")
