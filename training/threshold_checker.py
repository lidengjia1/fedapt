"""
阈值条件检测器
用于判断客户端是否应该停止蒸馏训练
"""
import torch
import torch.nn.functional as F
import numpy as np


class ThresholdChecker:
    """阈值条件检测器"""
    
    def __init__(self, 
                 acc_fluctuation_threshold=0.02,  # 准确率波动阈值 2%
                 cosine_sim_threshold=0.15,       # 余弦相似度阈值
                 stable_rounds=3):                # 需要连续稳定的轮数
        """
        Args:
            acc_fluctuation_threshold: 准确率波动阈值
            cosine_sim_threshold: X_r 与 X_s 的余弦相似度阈值
            stable_rounds: 连续稳定轮数要求
        """
        self.acc_threshold = acc_fluctuation_threshold
        self.cosine_threshold = cosine_sim_threshold
        self.stable_rounds = stable_rounds
        
        # 存储每个客户端的历史记录
        self.history = {}
    
    def check(self, client_id, accuracy, X_r, X_s):
        """
        检查客户端是否满足停止条件
        
        Args:
            client_id: 客户端ID
            accuracy: 当前轮次的分类准确率
            X_r: 重建特征
            X_s: 可泛化特征 (残差)
        
        Returns:
            should_stop: 是否应该停止蒸馏
            metrics: 诊断指标
        """
        # 初始化历史记录
        if client_id not in self.history:
            self.history[client_id] = {
                'accuracies': [],
                'cosine_similarities': []
            }
        
        # 更新历史
        self.history[client_id]['accuracies'].append(accuracy)
        
        # 计算余弦相似度
        cosine_sim = self._compute_cosine_similarity(X_r, X_s)
        self.history[client_id]['cosine_similarities'].append(cosine_sim)
        
        # 条件1：准确率连续3轮波动 < 2%
        acc_stable = self._check_accuracy_stability(client_id)
        
        # 条件2：余弦相似度 < 0.15
        low_similarity = cosine_sim < self.cosine_threshold
        
        # 综合判断
        should_stop = acc_stable and low_similarity
        
        metrics = {
            'accuracy': accuracy,
            'acc_stable': acc_stable,
            'cosine_similarity': cosine_sim,
            'low_similarity': low_similarity,
            'should_stop': should_stop
        }
        
        return should_stop, metrics
    
    def _compute_cosine_similarity(self, X_r, X_s):
        """
        计算 X_r 和 X_s 之间的平均余弦相似度
        
        Args:
            X_r: 重建特征 (batch_size, feature_dim)
            X_s: 残差特征 (batch_size, feature_dim)
        """
        # 确保是 Tensor
        if not isinstance(X_r, torch.Tensor):
            X_r = torch.tensor(X_r, dtype=torch.float32)
        if not isinstance(X_s, torch.Tensor):
            X_s = torch.tensor(X_s, dtype=torch.float32)
        
        # 计算余弦相似度
        cosine_sim = F.cosine_similarity(X_r, X_s, dim=1)
        
        # 返回平均值
        return cosine_sim.abs().mean().item()
    
    def _check_accuracy_stability(self, client_id):
        """
        检查准确率是否稳定（连续 stable_rounds 轮波动 < threshold）
        """
        accuracies = self.history[client_id]['accuracies']
        
        # 如果轮数不足，返回 False
        if len(accuracies) < self.stable_rounds:
            return False
        
        # 检查最近 stable_rounds 轮的波动
        recent_acc = accuracies[-self.stable_rounds:]
        fluctuation = max(recent_acc) - min(recent_acc)
        
        return fluctuation < self.acc_threshold
    
    def reset_client(self, client_id):
        """重置客户端历史记录"""
        if client_id in self.history:
            self.history[client_id] = {
                'accuracies': [],
                'cosine_similarities': []
            }
    
    def get_client_status(self, client_id):
        """获取客户端当前状态"""
        if client_id not in self.history:
            return None
        
        history = self.history[client_id]
        if len(history['accuracies']) == 0:
            return None
        
        return {
            'rounds_trained': len(history['accuracies']),
            'current_accuracy': history['accuracies'][-1],
            'avg_cosine_sim': np.mean(history['cosine_similarities'][-self.stable_rounds:])
                if len(history['cosine_similarities']) >= self.stable_rounds else None,
            'acc_trend': history['accuracies'][-5:] if len(history['accuracies']) >= 5 else history['accuracies']
        }
