"""
单数据集实验脚本 - 用于测试框架
"""
import torch
import numpy as np
import json
import logging
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from config.base_config import BaseConfig
from config.model_configs import get_model_config
from utils.data_loader import CreditDataLoader
from utils.partitioner import DataPartitioner
from models.classifier import create_classifier
from baselines.baseline_trainer import BaselineTrainer
from training.stage1_distillation import Stage1Distillation
from training.stage2_classification import Stage2Classification
from utils.metrics import evaluate_model, print_metrics, compare_methods
from utils.visualization import (
    plot_training_curves, plot_method_comparison,
    create_results_directory
)
from utils.setup_utils import set_random_seed


def setup_logging():
    """Setup logging configuration"""
    # 确保logs目录存在
    Path('results/logs').mkdir(parents=True, exist_ok=True)
    
    # 创建文件handler，设置为无缓冲模式
    file_handler = logging.FileHandler('results/logs/training.log', mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
    
    # 配置root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress tqdm output in log file
    logging.getLogger('tqdm').setLevel(logging.WARNING)
    
    # 强制刷新缓冲
    import sys
    sys.stdout.flush()
    sys.stderr.flush()


def run_single_experiment(dataset_name='australian', alpha=0.1, method='fedavg',
                         num_clients=10, learning_rate=0.001, partition_type='lda'):
    """
    运行单个实验
    
    Args:
        dataset_name: 数据集名称 ('australian', 'german', 'xinwang', 'uci')
        alpha: Dirichlet参数
        method: 方法名称 ('feddeproto', 'fedavg', 'fedprox', 'fedkf', 'fedfa', 'feddr+', 'fedtgp', 'fedfed')
        num_clients: 客户端数量
        learning_rate: 学习率
        partition_type: 数据划分方式 ('lda', 'label_skew', 'quantity_skew')
    """
    # Setup logging
    setup_logging()
    logger = logging.getLogger('Experiment')
    
    # 确保随机种子已设置（如果从main.py调用则已设置，否则这里设置）
    set_random_seed(42)
    
    logger.info("="*70)
    logger.info(f"Running Experiment: {method.upper()} on {dataset_name.upper()}")
    logger.info(f"Partition: {partition_type}" + (f" (α={alpha})" if partition_type == 'lda' else ""))
    logger.info(f"Clients: {num_clients}, LR: {learning_rate}")
    logger.info("="*70)
    
    # 配置
    config = BaseConfig()
    config.dataset_name = dataset_name
    config.alpha = alpha
    config.num_clients = num_clients
    config.learning_rate = float(learning_rate)  # 确保是float类型
    config.partition_type = partition_type
    
    # 验证学习率
    logger.info(f"配置学习率: {config.learning_rate} (类型: {type(config.learning_rate).__name__})")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    print(f"Using device: {device}\n")
    
    # 加载数据
    print("Loading data...")
    data_loader = CreditDataLoader(data_dir='data')
    X_train, X_test, y_train, y_test = data_loader.load_dataset(dataset_name)
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 数据分区
    print(f"\nPartitioning data with {partition_type}...")
    partitioner = DataPartitioner(num_clients)
    
    if partition_type == 'lda':
        client_indices = partitioner.partition_lda(y_train, alpha=alpha)
        print(f"Using LDA with α={alpha}")
    elif partition_type == 'label_skew':
        client_indices = partitioner.partition_label_skew(y_train)
        print(f"Using Label Skew")
    elif partition_type == 'quantity_skew':
        client_indices = partitioner.partition_label_skew(y_train)  # 暂时复用
        print(f"Using Quantity Skew")
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")
    
    # 统计分区信息
    for i, indices in enumerate(client_indices):
        labels = y_train[indices]
        class_0 = (labels == 0).sum()
        class_1 = (labels == 1).sum()
        print(f"Client {i}: {len(indices)} samples (Class 0: {class_0}, Class 1: {class_1})")
    
    # 创建客户端数据加载器
    from torch.utils.data import TensorDataset, DataLoader, Subset
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    client_loaders = [
        DataLoader(
            Subset(train_dataset, indices),
            batch_size=config.batch_size,
            shuffle=True
        )
        for indices in client_indices
    ]
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # 获取模型配置
    model_config = get_model_config(dataset_name)
    input_dim = X_train.shape[1]
    config.input_dim = input_dim  # 添加到配置中，FedDeProto需要
    
    # 创建模型
    print(f"\nCreating model for {dataset_name}...")
    model = create_classifier(
        classifier_type=model_config['classifier_type'],
        input_dim=input_dim,
        hidden_dims=model_config['classifier_hidden']
    )
    model.to(device)
    print(f"Model: {model_config['classifier_type']}")
    
    # 训练
    if method.lower() == 'feddeproto':
        logger.info("\n--- Running FedDeProto (Two-Stage Training) ---")
        
        from training.stage1_distillation import Stage1Distillation
        from training.stage2_classification import Stage2Classification
        
        # 阶段1: 特征蒸馏
        logger.info("Stage 1: Federated Feature Distillation")
        
        # 准备客户端数据字典格式
        clients_data = {}
        for client_id, loader in enumerate(client_loaders):
            X_list, y_list = [], []
            for batch_X, batch_y in loader:
                X_list.append(batch_X)
                y_list.append(batch_y)
            clients_data[client_id] = {
                'X': torch.cat(X_list, dim=0),
                'y': torch.cat(y_list, dim=0)
            }
        
        # 准备测试数据
        X_test_list, y_test_list = [], []
        for batch_X, batch_y in test_loader:
            X_test_list.append(batch_X)
            y_test_list.append(batch_y)
        test_data = {
            'X_test': torch.cat(X_test_list, dim=0),
            'y_test': torch.cat(y_test_list, dim=0)
        }
        
        # 运行阶段1
        stage1_trainer = Stage1Distillation(config, clients_data, test_data)
        stage1_history = stage1_trainer.train()
        shared_features = stage1_trainer.shared_features
        shared_labels = stage1_trainer.shared_labels
        
        logger.info(f"Stage 1 完成. 生成 {len(shared_features)} 个共享特征")
        
        # 阶段2: 联邦分类
        logger.info("Stage 2: Federated Classification")
        
        stage2_trainer = Stage2Classification(
            config=config,
            clients_data=clients_data,
            shared_features=shared_features,
            shared_labels=shared_labels,
            test_data=test_data,
            classifier_model=model
        )
        stage2_history = stage2_trainer.train()
        final_model = stage2_trainer.global_classifier
        
        # 合并历史记录
        history = {
            'loss': stage1_history.get('loss', []) + stage2_history.get('loss', []),
            'accuracy': stage1_history.get('accuracy', []) + stage2_history.get('accuracy', []),
            'stage1_rounds': len(stage1_history.get('loss', [])),
            'stage2_rounds': len(stage2_history.get('loss', [])),
            'stopped_clients': stage1_history.get('stopped_clients', 0)
        }
        
        logger.info("FedDeProto 两阶段训练完成")
        
    else:
        # 基准方法
        print(f"\n--- Running {method.upper()} Baseline ---")
        trainer = BaselineTrainer(
            model=model,
            client_data_loaders=client_loaders,
            test_loader=test_loader,
            config=config,
            method=method
        )
        
        num_rounds = config.T_d + config.T_r  # 总轮次
        history = trainer.train(num_rounds)
        final_model = trainer.global_model
    
    # 评估
    print("\n=== Final Evaluation ===")
    metrics = evaluate_model(final_model, test_loader, device)
    print_metrics(metrics, method_name=method.upper())
    
    # 保存结果
    results_dir = create_results_directory('results')
    
    result_data = {
        'dataset': dataset_name,
        'alpha': alpha,
        'method': method,
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'auc': float(metrics.get('auc', 0))
        },
        'history': {
            'loss': [float(x) for x in history.get('loss', [])],
            'accuracy': [float(x) for x in history.get('accuracy', [])]
        }
    }
    
    # 保存JSON
    result_file = results_dir / 'logs' / f'{dataset_name}_alpha{alpha}_{method}.json'
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=4)
    print(f"\nResults saved to {result_file}")
    
    # 绘制训练曲线
    if 'test_accuracy' in history:
        plot_path = results_dir / 'plots' / f'{dataset_name}_alpha{alpha}_{method}_training.png'
        plot_training_curves(
            {'accuracy': history['test_accuracy']},
            save_path=plot_path
        )
    
    return metrics, history


if __name__ == '__main__':
    # 测试单个实验
    metrics, history = run_single_experiment(
        dataset_name='australian',
        alpha=0.1,
        method='fedavg'
    )
    
    print("\nExperiment completed successfully!")
