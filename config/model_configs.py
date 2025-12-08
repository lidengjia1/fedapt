"""
FedDeProto - 模型配置文件
针对不同数据集的网络架构配置
"""

class ModelConfig:
    """模型配置字典"""
    
    CONFIGS = {
        'australian': {
            'input_dim': 14,  # 特征维度（Class列除外）
            'num_classes': 2,
            'latent_dim': 16,  # VAE潜在空间维度
            'classifier_type': 'mlp',  # 分类器类型
            
            # 编码器结构 (input_dim → hidden_layers → latent_dim)
            'encoder_hidden': [64, 32],
            
            # 解码器结构 (latent_dim → hidden_layers → input_dim)
            'decoder_hidden': [32, 64],
            
            # 判别器结构 (input_dim → hidden_layers → 1)
            'discriminator_hidden': [64, 32],
            
            # 分类器结构 (input_dim → hidden_layers → num_classes)
            'classifier_hidden': [32],
        },
        
        'german': {
            'input_dim': 20,  # 20个特征
            'num_classes': 2,
            'latent_dim': 16,
            'classifier_type': 'mlp',
            'encoder_hidden': [128, 64],
            'decoder_hidden': [64, 128],
            'discriminator_hidden': [128, 64],
            'classifier_hidden': [64, 32],
        },
        
        'xinwang': {
            'input_dim': 100,  # 100个特征（target列除外）
            'num_classes': 2,
            'latent_dim': 64,  # 大数据集用更大的潜在空间
            'classifier_type': 'mlp',
            'encoder_hidden': [256, 128],
            'decoder_hidden': [128, 256],
            'discriminator_hidden': [256, 128],
            'classifier_hidden': [128, 64, 32],
        },
        
        'uci': {
            'input_dim': None,  # 待动态检测
            'num_classes': 2,
            'latent_dim': 32,
            'classifier_type': 'mlp',
            'encoder_hidden': [128, 64],
            'decoder_hidden': [64, 128],
            'discriminator_hidden': [128, 64],
            'classifier_hidden': [64, 32],
        },
    }
    
    @staticmethod
    def get_config(dataset_name):
        """获取指定数据集的配置"""
        if dataset_name not in ModelConfig.CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return ModelConfig.CONFIGS[dataset_name].copy()
    
    @staticmethod
    def update_uci_config(input_dim):
        """根据实际读取的UCI数据更新配置"""
        ModelConfig.CONFIGS['uci']['input_dim'] = input_dim


# 便捷函数，与类方法保持一致
def get_model_config(dataset_name):
    """
    获取指定数据集的模型配置
    
    Args:
        dataset_name: 数据集名称 ('australian', 'german', 'xinwang', 'uci')
    
    Returns:
        dict: 模型配置字典
    """
    return ModelConfig.get_config(dataset_name)


# 为了向后兼容，也导出原来的配置字典
model_configs = ModelConfig.CONFIGS
