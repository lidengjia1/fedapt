"""
FedProx 基准方法 - 添加近端项约束
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import logging
from tqdm import tqdm
from federated.server import FederatedServer


class FedProxTrainer:
    """FedProx训练器 - 在本地目标函数中添加近端项"""
    
    def __init__(self, model, client_data_loaders, test_loader, config):
        self.global_model = model
        self.client_loaders = client_data_loaders
        self.test_loader = test_loader
        self.config = config
        
        self.num_clients = len(client_data_loaders)
        self.device = config.device
        self.mu = config.fedprox_mu  # 近端项系数
        
        self.server = FederatedServer(
            global_model=model,
            aggregation_method='fedavg'
        )
        
        # 初始化logger
        self.logger = logging.getLogger('FedProx_Trainer')
        
        self.history = {
            'loss': [],
            'accuracy': []
        }
    
    def proximal_term(self, local_model, global_model):
        """计算近端项 ||w - w_global||^2"""
        proximal_loss = 0.0
        for local_param, global_param in zip(local_model.parameters(), 
                                             global_model.parameters()):
            proximal_loss += torch.sum((local_param - global_param) ** 2)
        return proximal_loss
    
    def train_client(self, client_id, global_model):
        """训练单个客户端 - 添加近端正则化"""
        local_model = copy.deepcopy(global_model)
        local_model.to(self.device)
        local_model.train()
        
        # 保存全局模型副本用于计算近端项
        global_model_copy = copy.deepcopy(global_model)
        global_model_copy.to(self.device)
        for param in global_model_copy.parameters():
            param.requires_grad = False
        
        optimizer = torch.optim.Adam(
            local_model.parameters(),
            lr=self.config.learning_rate
        )
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0
        num_batches = 0
        
        for epoch in range(self.config.local_epochs):
            for batch_X, batch_y in self.client_loaders[client_id]:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # 分类损失
                outputs = local_model(batch_X)
                ce_loss = criterion(outputs, batch_y)
                
                # 近端项
                prox_loss = self.proximal_term(local_model, global_model_copy)
                
                # 总损失
                loss = ce_loss + (self.mu / 2) * prox_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
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
        
        return correct / total
    
    def train(self, num_rounds):
        """完整训练流程"""
        self.logger.info("Starting FedProx Training")
        self.logger.info(f"Number of clients: {self.num_clients}")
        self.logger.info(f"Number of rounds: {num_rounds}")
        self.logger.info(f"Proximal term coefficient (μ): {self.mu}")
        
        pbar = tqdm(range(num_rounds), desc="FedProx Training")
        
        for round_idx in pbar:
            selected_clients = range(self.num_clients)
            
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
            
            self.server.aggregate(client_models, client_weights)
            
            test_accuracy = self.evaluate()
            avg_loss = sum(round_losses) / len(round_losses)
            
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
        
        self.logger.info("FedProx Training Completed!")
        print(f"Final Test Accuracy: {self.history['test_accuracy'][-1]:.4f}")
        print(f"{'='*50}\n")
        
        return self.history
