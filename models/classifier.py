"""
分类器模型
用于第二阶段的联邦学习
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """多层感知机分类器"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], num_classes=2, dropout=0.3):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            num_classes: 类别数
            dropout: Dropout 比率
        """
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # 保存隐藏层作为特征提取器
        self.feature_extractor = nn.Sequential(*layers[:-1])  # 除最后一层
        self.classifier_layer = layers[-1]  # 最后的分类层
    
    def forward(self, x):
        return self.network(x)
    
    def get_features(self, x):
        """提取中间特征（用于FedFA, FedDr+等方法）"""
        return self.feature_extractor(x)


class ResidualClassifier(nn.Module):
    """用于残差特征 X_s 的分类器（蒸馏模型中使用）"""
    
    def __init__(self, input_dim, num_classes=2):
        super(ResidualClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x_residual):
        """
        Args:
            x_residual: 残差特征 X_s = X - X_r
        """
        x = F.relu(self.bn1(self.fc1(x_residual)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits
    
    def get_features(self, x_residual):
        """提取中间特征（用于FedFA, FedDr+等方法）"""
        x = F.relu(self.bn1(self.fc1(x_residual)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        return x


def create_classifier(classifier_type, input_dim, hidden_dims=None, num_classes=2):
    """
    创建分类器
    
    Args:
        classifier_type: 分类器类型 ('mlp' 或 'residual')
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表（可选）
        num_classes: 类别数
    
    Returns:
        分类器模型
    """
    if classifier_type == 'mlp':
        if hidden_dims is None:
            hidden_dims = [64, 32]
        return MLPClassifier(input_dim, hidden_dims=hidden_dims, num_classes=num_classes)
    elif classifier_type == 'residual':
        return ResidualClassifier(input_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
