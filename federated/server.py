"""
联邦学习服务器
负责模型聚合和分发
"""
import torch
import copy
from collections import OrderedDict


class FederatedServer:
    """联邦学习服务器"""
    
    def __init__(self, global_model, aggregation_strategy='fedavg'):
        """
        Args:
            global_model: 全局模型
            aggregation_strategy: 聚合策略 ('fedavg', 'adaptive', 'fedprox', etc.)
        """
        self.global_model = global_model
        self.aggregation_strategy = aggregation_strategy
        self.client_weights = {}  # 存储客户端权重
        self.stopped_clients = set()  # 已停止蒸馏的客户端
    
    def aggregate(self, client_models, client_weights=None, client_info=None):
        """
        聚合客户端模型
        
        Args:
            client_models: 客户端模型列表
            client_weights: 客户端权重（样本数）
            client_info: 客户端额外信息（如性能指标）
        
        Returns:
            aggregated_model: 聚合后的全局模型
        """
        if self.aggregation_strategy == 'fedavg':
            return self._fedavg(client_models, client_weights)
        elif self.aggregation_strategy == 'adaptive':
            return self._adaptive_aggregate(client_models, client_weights, client_info)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")
    
    def _fedavg(self, client_models, client_weights=None):
        """FedAvg 聚合"""
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        else:
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]
        
        global_dict = OrderedDict()
        
        # 初始化全局参数为0
        for key in client_models[0].state_dict().keys():
            global_dict[key] = torch.zeros_like(client_models[0].state_dict()[key], dtype=torch.float32)
        
        # 加权平均
        for client_model, weight in zip(client_models, client_weights):
            client_dict = client_model.state_dict()
            for key in global_dict.keys():
                # 转换为float32进行计算,避免类型冲突
                param = client_dict[key].float() if client_dict[key].dtype != torch.float32 else client_dict[key]
                global_dict[key] += weight * param
        
        # 转换回原始类型并更新全局模型
        for key in global_dict.keys():
            original_dtype = client_models[0].state_dict()[key].dtype
            global_dict[key] = global_dict[key].to(original_dtype)
        
        aggregated_model = copy.deepcopy(self.global_model)
        aggregated_model.load_state_dict(global_dict)
        
        return aggregated_model
    
    def _adaptive_aggregate(self, client_models, client_weights, client_info):
        """
        自适应权重衰减聚合
        用于第一阶段蒸馏，对已停止的客户端降低权重
        """
        if client_weights is None:
            client_weights = [1.0] * len(client_models)
        
        # 对已停止蒸馏的客户端应用权重衰减
        adaptive_weights = []
        for i, (client_id, info) in enumerate(client_info.items()):
            weight = client_weights[i]
            
            if client_id in self.stopped_clients:
                # 计算衰减因子
                decay_factor = self._compute_decay_factor(info)
                weight *= decay_factor
            
            adaptive_weights.append(weight)
        
        # 归一化权重
        total_weight = sum(adaptive_weights)
        adaptive_weights = [w / total_weight for w in adaptive_weights]
        
        # 使用自适应权重进行聚合
        return self._fedavg(client_models, adaptive_weights)
    
    def _compute_decay_factor(self, client_info, lambda_decay=0.5):
        """
        计算权重衰减因子
        
        Args:
            client_info: 客户端信息（包含性能变化）
            lambda_decay: 衰减强度超参数
        
        Returns:
            decay_factor: 衰减因子 exp(-λ * max(0, Δ_perf))
        """
        delta_perf = client_info.get('delta_perf', 0.0)
        decay_factor = torch.exp(torch.tensor(-lambda_decay * max(0, delta_perf)))
        return decay_factor.item()
    
    def mark_client_stopped(self, client_id):
        """标记客户端已停止蒸馏"""
        self.stopped_clients.add(client_id)
    
    def broadcast_model(self):
        """广播全局模型给客户端"""
        return copy.deepcopy(self.global_model)
    
    def get_model_parameters(self):
        """获取全局模型参数"""
        return self.global_model.state_dict()
