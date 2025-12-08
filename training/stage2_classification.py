"""
第二阶段：基于共享数据集的联邦分类训练
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import copy
import numpy as np
import logging
from tqdm import tqdm

from models.classifier import create_classifier
from federated.server import FederatedServer
from federated.aggregation import get_aggregator


class Stage2Classification:
    """第二阶段：联邦分类训练"""
    
    def __init__(self, config, clients_data, shared_features, shared_labels, test_data):
        """
        Args:
            config: 配置对象
            clients_data: 客户端本地数据
            shared_features: 全局共享特征
            shared_labels: 全局共享标签
            test_data: 测试数据
        """
        self.config = config
        self.clients_data = clients_data
        self.shared_features = shared_features
        self.shared_labels = shared_labels
        self.test_data = test_data
        self.num_clients = len(clients_data)
        
        # 初始化全局分类器
        self.global_classifier = create_classifier(
            dataset_name=config.dataset_name,
            input_dim=config.input_dim,
            num_classes=config.num_classes
        )
        
        # 初始化服务器
        self.server = FederatedServer(
            global_model=self.global_classifier,
            aggregation_strategy=config.aggregation_strategy
        )
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_classifier.to(self.device)
        
        # 初始化logger
        self.logger = logging.getLogger('Stage2_Classification')
        
        # 创建共享数据集
        self.shared_dataset = TensorDataset(
            torch.FloatTensor(shared_features),
            torch.LongTensor(shared_labels)
        )
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'test_acc': []
        }
    
    def train(self):
        """执行第二阶段训练"""
        self.logger.info("Stage 2: Federated Classification with Shared Dataset")
        
        best_test_acc = 0.0
        pbar = tqdm(range(self.config.num_rounds_stage2), desc="Stage 2")
        
        for round_idx in pbar:
            # 选择客户端
            selected_clients = self._select_clients(round_idx)
            
            # 客户端训练
            client_models = []
            client_weights = []
            
            for client_id in selected_clients:
                client_model, client_weight = self._train_client(client_id)
                client_models.append(client_model)
                client_weights.append(client_weight)
            
            # 服务器聚合
            self.global_classifier = self.server.aggregate(
                client_models, client_weights
            )
            
            # 评估
            test_acc = self._evaluate(round_idx)
            pbar.set_postfix({'Acc': f'{test_acc:.4f}'})
            
            # 保存最佳模型
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                self.best_model = copy.deepcopy(self.global_classifier)
        
        self.logger.info(f"Best Test Accuracy: {best_test_acc:.4f}")
        return self.train_history
    
    def _select_clients(self, round_idx):
        """选择参与训练的客户端"""
        # 简化实现：所有客户端都参与
        return list(range(self.num_clients))
    
    def _train_client(self, client_id):
        """
        训练单个客户端
        
        混合本地数据和全局共享数据
        """
        # 获取本地数据
        X_local = self.clients_data[client_id]['X']
        y_local = self.clients_data[client_id]['y']
        
        local_dataset = TensorDataset(
            torch.FloatTensor(X_local),
            torch.LongTensor(y_local)
        )
        
        # 混合本地数据和共享数据
        mixed_dataset = ConcatDataset([local_dataset, self.shared_dataset])
        
        dataloader = DataLoader(
            mixed_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # 创建客户端模型
        client_model = copy.deepcopy(self.global_classifier)
        client_model.to(self.device)
        client_model.train()
        
        # 优化器
        optimizer = optim.Adam(
            client_model.parameters(),
            lr=self.config.learning_rate
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # 本地训练
        for epoch in range(self.config.local_epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # 前向传播
                logits = client_model(batch_X)
                loss = criterion(logits, batch_y)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        client_weight = len(X_local)
        return client_model, client_weight
    
    def _evaluate(self, round_idx):
        """评估全局模型"""
        self.global_classifier.eval()
        
        test_dataset = TensorDataset(
            torch.FloatTensor(self.test_data['X_test']),
            torch.LongTensor(self.test_data['y_test'])
        )
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                logits = self.global_classifier(batch_X)
                predictions = torch.argmax(logits, dim=1)
                
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
        
        test_acc = correct / total
        self.train_history['test_acc'].append(test_acc)
        
        print(f"Test Accuracy: {test_acc:.4f}")
        return test_acc
