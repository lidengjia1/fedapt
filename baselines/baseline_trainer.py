"""
所有基准方法的统一训练器
"""
import torch
import torch.nn as nn
import copy
import logging
from tqdm import tqdm
from federated.aggregation import (
    FedAvgAggregator, FedProxAggregator, FedKFAggregator, 
    FedFAAggregator, FedDrPlusAggregator,
    FedTGPAggregator, FedFedAggregator
)


class BaselineTrainer:
    """统一的基准方法训练器"""
    
    def __init__(self, model, client_data_loaders, test_loader, config, method='fedavg'):
        """
        Args:
            method: 'fedavg', 'fedkf', 'fedfa', 'feddr+', 'fedtgp', 'fedfed'
        """
        self.global_model = model
        self.client_loaders = client_data_loaders
        self.test_loader = test_loader
        self.config = config
        self.method = method.lower()
        
        self.num_clients = len(client_data_loaders)
        self.device = config.device
        
        # 初始化聚合器
        self._init_aggregator()
        
        # 初始化logger
        self.logger = logging.getLogger(f'{self.method.upper()}_Trainer')
        
        self.history = {
            'loss': [],
            'accuracy': []
        }
    
    def _init_aggregator(self):
        """根据方法初始化对应的聚合器"""
        if self.method == 'fedavg':
            self.aggregator = FedAvgAggregator()
        elif self.method == 'fedprox':
            self.aggregator = FedProxAggregator()
        elif self.method == 'fedkf':
            self.aggregator = FedKFAggregator(
                process_noise=0.01,
                measurement_noise=0.1
            )
        elif self.method == 'fedfa':
            self.aggregator = FedFAAggregator(alignment_weight=0.1)
        elif self.method == 'feddr+':
            self.aggregator = FedDrPlusAggregator(
                num_classes=2,
                prototype_momentum=0.9
            )
        elif self.method == 'fedtgp':
            self.aggregator = FedTGPAggregator(
                history_size=5,
                prediction_weight=0.3
            )
        elif self.method == 'fedfed':
            self.aggregator = FedFedAggregator(distillation_temperature=3.0)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def train_client(self, client_id, global_model):
        """训练单个客户端"""
        local_model = copy.deepcopy(global_model)
        local_model.to(self.device)
        local_model.train()
        
        # 保存全局模型参数（用于FedProx）
        if self.method == 'fedprox':
            global_params = {name: param.clone().detach() 
                           for name, param in global_model.state_dict().items()}
        
        optimizer = torch.optim.Adam(
            local_model.parameters(),
            lr=self.config.learning_rate
        )
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0
        num_batches = 0
        
        # 收集特征和原型（用于某些方法）
        features_list = []
        labels_list = []
        prototypes = {}
        
        for epoch in range(self.config.local_epochs):
            for batch_X, batch_y in self.client_loaders[client_id]:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = local_model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # FedProx: 添加近端项 μ/2 * ||w - w_global||^2
                if self.method == 'fedprox':
                    proximal_term = 0.0
                    mu = 0.01  # 近端系数
                    for name, param in local_model.named_parameters():
                        if name in global_params:
                            proximal_term += ((param - global_params[name]) ** 2).sum()
                    loss += (mu / 2) * proximal_term
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # 收集特征（用于FedFA, FedDr+, FedFed等方法）
                if self.method in ['fedfa', 'feddr+', 'fedfed']:
                    with torch.no_grad():
                        # 假设模型有中间特征输出
                        if hasattr(local_model, 'get_features'):
                            features = local_model.get_features(batch_X)
                            features_list.append(features.cpu())
                            labels_list.append(batch_y.cpu())
        
        # FedDr+: 计算类原型
        if self.method == 'feddr+' and features_list:
            all_features = torch.cat(features_list, dim=0)
            all_labels = torch.cat(labels_list, dim=0)
            
            # 计算每个类别的原型（特征均值）
            unique_labels = torch.unique(all_labels)
            for label in unique_labels:
                mask = (all_labels == label)
                if mask.sum() > 0:
                    class_features = all_features[mask]
                    prototypes[f'class_{label.item()}'] = class_features.mean(dim=0)
        
        avg_loss = total_loss / num_batches
        num_samples = len(self.client_loaders[client_id].dataset)
        
        result = {
            'model': local_model.state_dict(),
            'num_samples': num_samples,
            'loss': avg_loss
        }
        
        # 添加额外信息
        if features_list:
            result['features'] = torch.cat(features_list, dim=0)
            result['labels'] = torch.cat(labels_list, dim=0)
        
        if prototypes:
            result['prototypes'] = prototypes
        
        return result
    
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
    
    def aggregate_models(self, client_results):
        """根据方法进行模型聚合"""
        # 从state_dict重建模型对象
        client_models = []
        for r in client_results:
            model = copy.deepcopy(self.global_model)
            model.load_state_dict(r['model'])
            client_models.append(model)
        
        client_weights = [r['num_samples'] for r in client_results]
        
        # 提取额外信息
        client_features = [r.get('features', None) for r in client_results]
        client_prototypes = [r.get('prototypes', None) for r in client_results]
        
        # 统一使用聚合器
        if self.method in ['fedavg', 'fedprox']:
            aggregated_model = self.aggregator.aggregate(client_models, client_weights)
        elif self.method == 'fedkf':
            aggregated_model = self.aggregator.aggregate(client_models, client_weights)
        elif self.method == 'fedfa':
            aggregated_model = self.aggregator.aggregate(
                client_models, client_weights, client_features=client_features
            )
        elif self.method == 'feddr+':
            aggregated_model = self.aggregator.aggregate(
                client_models, client_weights, client_prototypes=client_prototypes
            )
        elif self.method == 'fedtgp':
            aggregated_model = self.aggregator.aggregate(client_models, client_weights)
        elif self.method == 'fedfed':
            aggregated_model = self.aggregator.aggregate(client_models, client_weights)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.global_model.load_state_dict(aggregated_model.state_dict())
    
    def train(self, num_rounds):
        """完整训练流程"""
        self.logger.info(f"Starting {self.method.upper()} Training")
        self.logger.info(f"Number of clients: {self.num_clients}")
        self.logger.info(f"Number of rounds: {num_rounds}")
        
        pbar = tqdm(range(num_rounds), desc=f"{self.method.upper()} Training")
        
        for round_idx in pbar:
            selected_clients = range(self.num_clients)
            
            client_results = []
            
            for client_id in selected_clients:
                result = self.train_client(client_id, self.global_model)
                client_results.append(result)
            
            # 聚合
            self.aggregate_models(client_results)
            
            # 评估
            test_accuracy = self.evaluate()
            avg_loss = sum(r['loss'] for r in client_results) / len(client_results)
            
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
        
        self.logger.info(f"{self.method.upper()} Training Completed!")
        self.logger.info(f"Final Test Accuracy: {self.history['accuracy'][-1]:.4f}")
        
        return self.history
