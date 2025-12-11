"""
聚合策略模块
实现多种联邦学习聚合算法
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict
import numpy as np


class FedAvgAggregator:
    """FedAvg 聚合器 - 标准加权平均"""
    
    @staticmethod
    def aggregate(client_models, client_weights=None):
        """
        标准 FedAvg 聚合：按样本数加权平均
        
        Args:
            client_models: 客户端模型列表
            client_weights: 客户端权重（通常是样本数）
        """
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        else:
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]
        
        global_dict = OrderedDict()
        for key in client_models[0].state_dict().keys():
            global_dict[key] = torch.zeros_like(client_models[0].state_dict()[key]).float()
        
        for client_model, weight in zip(client_models, client_weights):
            client_dict = client_model.state_dict()
            for key in global_dict.keys():
                global_dict[key] += weight * client_dict[key].float()
        
        global_model = copy.deepcopy(client_models[0])
        global_model.load_state_dict(global_dict)
        return global_model


class FedProxAggregator:
    """FedProx 聚合器 - 带近端项正则化（客户端实现）"""
    
    @staticmethod
    def aggregate(client_models, client_weights=None):
        """
        FedProx 聚合：服务端与FedAvg相同，近端项在客户端训练时添加
        近端项: ||w - w_global||^2
        """
        return FedAvgAggregator.aggregate(client_models, client_weights)


class FedKFAggregator:
    """FedKF - 卡尔曼滤波聚合器（动态调整聚合权重）"""
    
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        """
        Args:
            process_noise: 过程噪声（模型更新的不确定性）
            measurement_noise: 测量噪声（客户端模型的不确定性）
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state_mean = None  # 状态均值（全局模型参数）
        self.state_cov = None   # 状态协方差（参数不确定性）
    
    def aggregate(self, client_models, client_weights=None):
        """
        使用卡尔曼滤波聚合：考虑历史信息和不确定性
        """
        # 获取设备（使用第一个模型的设备）
        device = next(client_models[0].parameters()).device
        
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        else:
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]
        
        global_dict = OrderedDict()
        
        # 第一次：直接平均初始化
        if self.state_mean is None:
            for key in client_models[0].state_dict().keys():
                params = torch.stack([m.state_dict()[key].float().to(device) for m in client_models])
                weights_tensor = torch.tensor(client_weights, device=device).view(-1, *([1]*(params.dim()-1)))
                global_dict[key] = torch.sum(params * weights_tensor, dim=0)
            
            global_model = copy.deepcopy(client_models[0])
            global_model.load_state_dict(global_dict)
            
            # 初始化状态
            self.state_mean = {k: v.clone() for k, v in global_dict.items()}
            self.state_cov = {k: torch.ones_like(v) * self.process_noise for k, v in global_dict.items()}
            
            return global_model
        
        # 卡尔曼滤波更新
        for key in client_models[0].state_dict().keys():
            # 预测步骤：状态预测 + 协方差预测
            pred_mean = self.state_mean[key].to(device)
            pred_cov = self.state_cov[key].to(device) + self.process_noise
            
            # 测量：客户端模型参数的加权平均
            measurements = torch.stack([m.state_dict()[key].float().to(device) for m in client_models])
            weights_tensor = torch.tensor(client_weights, device=device).view(-1, *([1]*(measurements.dim()-1)))
            measurement = torch.sum(measurements * weights_tensor, dim=0)
            
            # 卡尔曼增益：K = P_pred / (P_pred + R)
            kalman_gain = pred_cov / (pred_cov + self.measurement_noise)
            
            # 更新步骤
            self.state_mean[key] = pred_mean + kalman_gain * (measurement - pred_mean)
            self.state_cov[key] = (1 - kalman_gain) * pred_cov
            
            global_dict[key] = self.state_mean[key]
        
        global_model = copy.deepcopy(client_models[0])
        global_model.load_state_dict(global_dict)
        return global_model


class FedFAAggregator:
    """FedFA - 特征对齐聚合器（对齐不同客户端的特征分布）"""
    
    def __init__(self, alignment_weight=0.1):
        """
        Args:
            alignment_weight: 特征对齐的权重系数
        """
        self.alignment_weight = alignment_weight
        self.global_feature_stats = None  # 全局特征统计
    
    def aggregate(self, client_models, client_weights=None, client_features=None):
        """
        FedFA聚合：根据特征对齐程度调整聚合权重
        
        Args:
            client_features: 客户端特征 [(features, labels), ...]
        """
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        else:
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]
        
        # 如果有特征信息，根据特征相似度调整权重
        if client_features is not None and len(client_features) > 0:
            valid_features = [f for f in client_features if f is not None]
            
            if len(valid_features) > 1:
                # 计算特征中心
                feature_center = torch.cat(valid_features).mean(dim=0)
                
                # 计算每个客户端与中心的相似度
                similarities = []
                for features in client_features:
                    if features is not None:
                        # 余弦相似度
                        feat_mean = features.mean(dim=0)
                        sim = torch.nn.functional.cosine_similarity(
                            feat_mean.unsqueeze(0), 
                            feature_center.unsqueeze(0)
                        ).item()
                        similarities.append(sim)
                    else:
                        similarities.append(0.5)  # 默认相似度
                
                # 根据相似度调整权重：相似度高的客户端获得更高权重
                adjusted_weights = []
                for w, sim in zip(client_weights, similarities):
                    # 权重调整：w' = w * (1 + α * sim)
                    adjusted_w = w * (1.0 + self.alignment_weight * sim)
                    adjusted_weights.append(adjusted_w)
                
                # 重新归一化
                total_w = sum(adjusted_weights)
                client_weights = [w / total_w for w in adjusted_weights]
        
        # 使用调整后的权重进行聚合
        global_model = FedAvgAggregator.aggregate(client_models, client_weights)
        
        return global_model


class FedDrPlusAggregator:
    """FedDr+ - 基于原型蒸馏的聚合器（使用类原型对齐）"""
    
    def __init__(self, num_classes=2, prototype_momentum=0.9):
        """
        Args:
            num_classes: 分类数量
            prototype_momentum: 原型更新动量
        """
        self.num_classes = num_classes
        self.momentum = prototype_momentum
        self.global_prototypes = None  # 全局类原型
    
    def aggregate(self, client_models, client_weights=None, client_prototypes=None):
        """
        FedDr+聚合：根据原型质量调整模型聚合权重
        
        Args:
            client_prototypes: 客户端原型 [{'class_0': proto0, 'class_1': proto1}, ...]
        """
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        else:
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]
        
        # 1. 原型聚合和质量评估
        if client_prototypes is not None and len(client_prototypes) > 0:
            # 计算全局原型（用于评估客户端原型质量）
            global_protos = {}
            for class_id in range(self.num_classes):
                class_protos = []
                for proto_dict in client_prototypes:
                    if proto_dict and f'class_{class_id}' in proto_dict:
                        class_protos.append(proto_dict[f'class_{class_id}'])
                
                if len(class_protos) > 0:
                    global_protos[f'class_{class_id}'] = torch.stack(class_protos).mean(dim=0)
            
            # 根据原型质量调整权重（使用当前轮的全局原型作为参考）
            if len(global_protos) > 0:
                # 使用当前计算的全局原型或历史全局原型作为参考
                reference_protos = self.global_prototypes if self.global_prototypes is not None else global_protos
                
                proto_scores = []
                for proto_dict in client_prototypes:
                    if proto_dict:
                        # 计算客户端原型与参考原型的一致性
                        score = 0.0
                        count = 0
                        for class_key in global_protos:
                            if class_key in proto_dict and class_key in reference_protos:
                                # 余弦相似度
                                sim = torch.nn.functional.cosine_similarity(
                                    proto_dict[class_key].unsqueeze(0),
                                    reference_protos[class_key].unsqueeze(0)
                                ).item()
                                score += sim
                                count += 1
                        proto_scores.append(score / count if count > 0 else 0.5)
                    else:
                        proto_scores.append(0.5)
                
                # 调整权重：原型质量高的客户端获得更高权重
                adjusted_weights = []
                for w, score in zip(client_weights, proto_scores):
                    adjusted_w = w * (1.0 + (1.0 - self.momentum) * score)
                    adjusted_weights.append(adjusted_w)
                
                # 重新归一化
                total_w = sum(adjusted_weights)
                client_weights = [w / total_w for w in adjusted_weights]
            
            # 更新全局原型
            if self.global_prototypes is None:
                self.global_prototypes = global_protos
            else:
                for class_key in global_protos:
                    if class_key in self.global_prototypes:
                        self.global_prototypes[class_key] = \
                            self.momentum * self.global_prototypes[class_key] + \
                            (1 - self.momentum) * global_protos[class_key]
                    else:
                        self.global_prototypes[class_key] = global_protos[class_key]
        
        # 2. 使用调整后的权重进行模型聚合
        global_model = FedAvgAggregator.aggregate(client_models, client_weights)
        
        return global_model


class FedTGPAggregator:
    """FedTGP - 时序梯度预测聚合器（利用历史梯度信息预测更新）"""
    
    def __init__(self, history_size=5, prediction_weight=0.3):
        """
        Args:
            history_size: 保存的历史梯度轮数
            prediction_weight: 预测梯度的权重
        """
        self.history_size = history_size
        self.prediction_weight = prediction_weight
        self.gradient_history = []  # 存储历史梯度
        self.prev_global_state = None  # 上一轮全局模型
    
    def aggregate(self, client_models, client_weights=None):
        """
        FedTGP聚合：利用历史梯度预测未来更新
        """
        # 1. 基础聚合
        global_model = FedAvgAggregator.aggregate(client_models, client_weights)
        
        # 2. 计算当前轮次的平均梯度
        if self.prev_global_state is not None:
            current_gradient = {}
            global_state = global_model.state_dict()
            
            for name in global_state:
                if name in self.prev_global_state:
                    current_gradient[name] = global_state[name] - self.prev_global_state[name]
            
            # 3. 基于历史梯度进行预测
            if len(self.gradient_history) >= 1:
                # 简单线性预测：预测 = 当前梯度 + α * (当前梯度 - 上次梯度)
                prev_gradient = self.gradient_history[-1]
                
                updated_state = global_model.state_dict()
                for name in current_gradient:
                    if name in prev_gradient:
                        # 梯度动量：当前梯度 + α * (当前梯度 - 上次梯度)
                        gradient_change = current_gradient[name] - prev_gradient[name]
                        predicted_update = self.prediction_weight * gradient_change
                        updated_state[name] = updated_state[name] + predicted_update
                
                global_model.load_state_dict(updated_state)
            
            # 记录当前梯度
            self.gradient_history.append(current_gradient)
            
            # 保持固定历史长度
            if len(self.gradient_history) > self.history_size:
                self.gradient_history.pop(0)
        
        # 保存当前全局状态
        self.prev_global_state = {k: v.clone() for k, v in global_model.state_dict().items()}
        
        return global_model


class FedFedAggregator:
    """FedFed - 联邦特征蒸馏聚合器（通过知识蒸馏对齐客户端）"""
    
    def __init__(self, distillation_temperature=3.0):
        """
        Args:
            distillation_temperature: 蒸馏温度（用于客户端训练）
        """
        self.temperature = distillation_temperature
    
    def aggregate(self, client_models, client_weights=None):
        """
        FedFed聚合：标准聚合（蒸馏主要在客户端训练时进行）
        
        注意：蒸馏主要在客户端训练时进行，服务端只做标准聚合
        """
        # 服务端进行标准聚合
        global_model = FedAvgAggregator.aggregate(client_models, client_weights)
        
        return global_model


def get_aggregator(aggregation_type='fedavg', **kwargs):
    """
    工厂函数：获取聚合器
    
    Args:
        aggregation_type: 聚合类型
        **kwargs: 聚合器特定参数
    """
    aggregators = {
        'fedavg': FedAvgAggregator,
        'fedprox': FedProxAggregator,
        'fedkf': FedKFAggregator,
        'fedfa': FedFAAggregator,
        'feddr+': FedDrPlusAggregator,
        'fedtgp': FedTGPAggregator,
        'fedfed': FedFedAggregator
    }
    
    if aggregation_type.lower() not in aggregators:
        raise ValueError(f"Unknown aggregation type: {aggregation_type}")
    
    aggregator_class = aggregators[aggregation_type.lower()]
    
    # 所有聚合器都返回实例
    return aggregator_class(**kwargs) if kwargs else aggregator_class()
