"""
Focal Loss - 用于处理类别不平衡问题
Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss实现
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: 平衡因子，用于平衡正负样本
        gamma: 聚焦参数，减少易分样本的权重
        reduction: 'mean', 'sum' or 'none'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets, class_weights=None):
        """
        Args:
            inputs: 预测logits, shape (N, C)
            targets: 真实标签, shape (N,)
            class_weights: 类别权重, shape (C,) or None
        
        Returns:
            loss: Focal Loss值
        """
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=class_weights)
        
        # 计算预测概率
        p = torch.exp(-ce_loss)  # p_t
        
        # Focal Loss
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    结合类别权重的Focal Loss
    适用于严重类别不平衡的场景
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha  # shape: (num_classes,)
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: shape (N, C)
            targets: shape (N,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        
        # 应用alpha权重
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - p) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(loss_type='ce', use_focal=False, alpha=0.25, gamma=2.0, class_weights=None):
    """
    获取损失函数
    
    Args:
        loss_type: 'ce' (CrossEntropy) or 'focal'
        use_focal: 是否使用Focal Loss
        alpha: Focal Loss的alpha参数
        gamma: Focal Loss的gamma参数
        class_weights: 类别权重
    
    Returns:
        损失函数
    """
    if use_focal or loss_type == 'focal':
        if class_weights is not None:
            # 将class_weights转换为alpha
            alpha_weights = torch.FloatTensor(class_weights)
            return WeightedFocalLoss(alpha=alpha_weights, gamma=gamma)
        else:
            return FocalLoss(alpha=alpha, gamma=gamma)
    else:
        return nn.CrossEntropyLoss(weight=class_weights)


def compute_enhanced_class_weights(y, imbalance_threshold=0.3):
    """
    计算增强的类别权重
    
    当类别不平衡严重时（minority_ratio < threshold），增强少数类权重
    
    Args:
        y: 标签数组
        imbalance_threshold: 不平衡阈值
    
    Returns:
        class_weights: numpy数组
    """
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    # 基础权重
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    
    # 检查不平衡程度
    class_counts = np.bincount(y)
    minority_ratio = min(class_counts) / max(class_counts)
    
    print(f"类别分布: {class_counts}, 不平衡比率: {1/minority_ratio:.2f}:1")
    
    if minority_ratio < imbalance_threshold:
        # 严重不平衡，增强少数类权重
        power = 1.0 + (imbalance_threshold - minority_ratio) * 2  # 1.0 ~ 1.6
        class_weights = class_weights ** power
        print(f"⚠️ 检测到严重不平衡（{1/minority_ratio:.1f}:1），增强少数类权重 (power={power:.2f})")
    
    print(f"类别权重: {class_weights}")
    return class_weights
