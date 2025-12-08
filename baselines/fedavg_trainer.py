"""
FedAvg 基准方法
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import logging
from tqdm import tqdm
from federated.server import FederatedServer


class FedAvgTrainer:
    """标准FedAvg训练器"""
    
    def __init__(self, model, client_data_loaders, test_loader, config):
        """
        Args:
            model: 全局模型
            client_data_loaders: 客户端数据加载器列表
            test_loader: 测试数据加载器
            config: 配置对象
        """
        self.global_model = model
        self.client_loaders = client_data_loaders
        self.test_loader = test_loader
        self.config = config
        
        self.num_clients = len(client_data_loaders)
        self.device = config.device
        
        # 初始化服务器
        self.server = FederatedServer(
            global_model=model,
            aggregation_method='fedavg'
        )
        
        # 初始化logger
        self.logger = logging.getLogger('FedAvg_Trainer')
        
        # 训练历史
        self.history = {
            'loss': [],
            'accuracy': []
        }
    
    def train_client(self, client_id, global_model):
        """训练单个客户端"""
        # 复制全局模型
        local_model = copy.deepcopy(global_model)
        local_model.to(self.device)
        local_model.train()
        
        optimizer = torch.optim.Adam(
            local_model.parameters(),
            lr=self.config.learning_rate
        )
        criterion = nn.CrossEntropyLoss()
        
        # 本地训练
        total_loss = 0
        num_batches = 0
        
        for epoch in range(self.config.local_epochs):
            for batch_X, batch_y in self.client_loaders[client_id]:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = local_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # 返回本地模型参数和样本数
        num_samples = len(self.client_loaders[client_id].dataset)
        
        return local_model.state_dict(), num_samples, avg_loss
    
    def evaluate(self):
        """评估全局模型"""
        self.global_model.eval()
        self.global_model.to(self.device)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.global_model(batch_X)
                predictions = torch.argmax(outputs, dim=1)
                
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
        
        accuracy = correct / total
        return accuracy
    
    def train(self, num_rounds):
        """完整训练流程"""
        self.logger.info("Starting FedAvg Training")
        self.logger.info(f"Number of clients: {self.num_clients}")
        self.logger.info(f"Number of rounds: {num_rounds}")
        self.logger.info(f"Local epochs: {self.config.local_epochs}")
        
        pbar = tqdm(range(num_rounds), desc="FedAvg Training")
        
        for round_idx in pbar:
            # 选择客户端（默认全部参与）
            selected_clients = range(self.num_clients)
            
            # 收集客户端模型
            client_models = []
            client_weights = []
            round_losses = []
            
            for client_id in selected_clients:
                model_params, num_samples, loss = self.train_client(
                    client_id, self.global_model
                )
                client_models.append(model_params)
                client_weights.append(num_samples)
                round_losses.append(loss)
            
            # 聚合
            self.server.aggregate(client_models, client_weights)
            
            # 评估
            test_accuracy = self.evaluate()
            avg_loss = sum(round_losses) / len(round_losses)
            
            # 记录历史
            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(test_accuracy)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{test_accuracy:.4f}'
            })
            
            # 记录到logger
            self.logger.info(f"Round {round_idx + 1}/{num_rounds} - "
                           f"Loss: {avg_loss:.4f}, Accuracy: {test_accuracy:.4f}")
        
        self.logger.info("FedAvg Training Completed!")
        self.logger.info(f"Final Test Accuracy: {self.history['accuracy'][-1]:.4f}")
        print("FedAvg Training Completed!")
        print(f"Final Test Accuracy: {self.history['test_accuracy'][-1]:.4f}")
        print(f"{'='*50}\n")
        
        return self.history
