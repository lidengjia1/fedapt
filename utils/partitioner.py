"""
Non-IID数据划分器
使用 Dirichlet 分布模拟数据异构性
"""
import numpy as np
from collections import defaultdict


class DataPartitioner:
    """数据划分器，支持多种 Non-IID 策略"""
    
    def __init__(self, num_clients=10, random_state=42):
        """
        Args:
            num_clients: 客户端数量
            random_state: 随机种子
        """
        self.num_clients = num_clients
        self.random_state = random_state
        np.random.seed(random_state)
    
    def partition_lda(self, y, alpha=0.1, X=None):
        """
        使用 Latent Dirichlet Allocation (LDA) 划分数据
        
        Args:
            y: 标签向量 (n_samples,)
            alpha: Dirichlet 分布参数，越小越不均衡
            X: 特征矩阵 (可选，为了兼容性保留)
        
        Returns:
            list: 客户端索引列表 [[idx1, idx2, ...], [idx3, idx4, ...], ...]
        """
        num_classes = len(np.unique(y))
        client_indices_list = []
        
        # 按类别分组
        class_indices = defaultdict(list)
        for idx, label in enumerate(y):
            class_indices[label].append(idx)
        
        # 为每个客户端从 Dirichlet 分布采样类别分布
        client_class_distributions = np.random.dirichlet([alpha] * num_classes, self.num_clients)
        
        # 计算每个客户端应分配的样本数
        total_samples = len(y)
        samples_per_client = total_samples // self.num_clients
        
        # 为每个客户端分配数据
        for client_id in range(self.num_clients):
            client_indices = []
            class_dist = client_class_distributions[client_id]
            
            # 根据类别分布分配样本
            for class_label in range(num_classes):
                n_samples_for_class = int(samples_per_client * class_dist[class_label])
                available_indices = class_indices[class_label]
                
                if len(available_indices) >= n_samples_for_class:
                    selected_indices = np.random.choice(
                        available_indices, 
                        size=n_samples_for_class, 
                        replace=False
                    )
                    client_indices.extend(selected_indices)
                    # 移除已分配的索引
                    class_indices[class_label] = [idx for idx in available_indices if idx not in selected_indices]
            
            # 如果样本不足，从剩余样本中随机补充
            if len(client_indices) < samples_per_client:
                remaining_indices = [idx for indices in class_indices.values() for idx in indices]
                if len(remaining_indices) > 0:
                    additional_samples = min(samples_per_client - len(client_indices), len(remaining_indices))
                    additional_indices = np.random.choice(remaining_indices, size=additional_samples, replace=False)
                    client_indices.extend(additional_indices)
            
            client_indices_list.append(list(client_indices))
        
        return client_indices_list
    
    def partition_label_skew(self, y, num_major_classes=2, X=None):
        """
        标签倾斜划分：每个客户端只包含少数类别
        
        Args:
            y: 标签向量
            num_major_classes: 每个客户端主要包含的类别数
            X: 特征矩阵 (可选)
        
        Returns:
            list: 客户端索引列表
        Returns:
            list: 客户端索引列表
        """
        num_classes = len(np.unique(y))
        client_indices_list = []
        
        # 为每个客户端随机分配类别
        for client_id in range(self.num_clients):
            selected_classes = np.random.choice(num_classes, num_major_classes, replace=False)
            indices = [i for i, label in enumerate(y) if label in selected_classes]
            
            # 随机抽样
            samples_per_client = len(indices) // (self.num_clients // num_classes) if num_classes > 0 else len(indices)
            if len(indices) > samples_per_client:
                indices = np.random.choice(indices, samples_per_client, replace=False)
            
            client_indices_list.append(list(indices))
        
        return client_indices_list
    
    def partition_quantity_skew(self, X, y, imbalance_ratio=0.9):
        """
        数量倾斜划分：某些客户端的数据量远大于其他客户端
        
        Args:
            imbalance_ratio: 主要类别占比
        """
        num_classes = len(np.unique(y))
        clients_data = {}
        
        total_samples = len(y)
        all_indices = list(range(total_samples))
        np.random.shuffle(all_indices)
        
        # 为每个客户端分配不同数量的样本
        samples_per_client = []
        remaining = total_samples
        for i in range(self.num_clients - 1):
            # 随机分配样本数，但保持一定的不均衡性
            max_samples = int(remaining * 0.5)
            min_samples = int(remaining * 0.05)
            n_samples = np.random.randint(min_samples, max_samples + 1)
            samples_per_client.append(n_samples)
            remaining -= n_samples
        samples_per_client.append(remaining)  # 最后一个客户端获得剩余样本
        
        start_idx = 0
        for client_id, n_samples in enumerate(samples_per_client):
            indices = all_indices[start_idx:start_idx + n_samples]
            
            clients_data[client_id] = {
                'X': X[indices],
                'y': y[indices],
                'indices': indices
            }
            start_idx += n_samples
        
        return clients_data
    
    def get_statistics(self, clients_data):
        """获取数据划分统计信息"""
        stats = {}
        for client_id, data in clients_data.items():
            y_client = data['y']
            unique, counts = np.unique(y_client, return_counts=True)
            stats[client_id] = {
                'total_samples': len(y_client),
                'class_distribution': dict(zip(unique.tolist(), counts.tolist()))
            }
        return stats


def partition_data(X, y, num_clients=10, partition_type='lda', alpha=0.1, **kwargs):
    """
    便捷函数：划分数据
    
    Args:
        X: 特征矩阵
        y: 标签向量
        num_clients: 客户端数量
        partition_type: 划分类型 ('lda', 'label_skew', 'quantity_skew')
        alpha: Dirichlet 参数（仅用于 LDA）
        **kwargs: 其他参数
    
    Returns:
        clients_data: 划分后的客户端数据
    """
    partitioner = DataPartitioner(num_clients=num_clients)
    
    if partition_type == 'lda':
        return partitioner.partition_lda(X, y, alpha=alpha)
    elif partition_type == 'label_skew':
        return partitioner.partition_label_skew(X, y, **kwargs)
    elif partition_type == 'quantity_skew':
        return partitioner.partition_quantity_skew(X, y, **kwargs)
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")
