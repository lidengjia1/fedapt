"""
FedDeProto: 联邦学习信用风险评估系统
主入口文件
"""
import argparse
import torch
import logging
from pathlib import Path

from experiments.run_single_dataset import run_single_experiment
from experiments.run_all_experiments import ExperimentRunner
from utils.setup_utils import initialize_experiment_environment, print_experiment_info


def main():
    # 配置基础logging（在初始化环境之前）
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='FedDeProto Credit Risk Assessment')
    
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'full', 'experiments'],
                       help='Experiment mode: single, full, or experiments (分组对照)')
    
    parser.add_argument('--dataset', type=str, default='australian',
                       choices=['australian', 'german', 'xinwang', 'uci'],
                       help='Dataset name')
    
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Dirichlet alpha for data partitioning')
    
    parser.add_argument('--method', type=str, default='fedavg',
                       choices=['feddeproto', 'fedavg', 'fedprox', 'fedkf', 
                               'fedfa', 'feddr+', 'fedtgp', 'fedfed'],
                       help='Training method')
    
    parser.add_argument('--num-clients', type=int, default=10,
                       help='Number of clients (5, 8, or 10)')
    
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.001,
                       dest='learning_rate',
                       help='Learning rate (0.001, 0.01, or 0.1)')
    
    parser.add_argument('--partition-type', type=str, default='lda',
                       choices=['lda', 'label_skew', 'quantity_skew'],
                       help='Data partition type')
    
    parser.add_argument('--epsilon', type=float, default=1.0,
                       help='Differential privacy epsilon')
    
    parser.add_argument('--groups', type=str, default='all',
                       help='实验组 (用于experiments模式): all, A, B, C, D, E, 或组合如 A,B')
    
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (-1 for CPU)')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    parser.add_argument('--no-clear', action='store_true',
                       help='⚠️ Do not clear results directory before running (preserve previous results)')
    
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompt and run experiments directly')
    
    args = parser.parse_args()
    
    # ⚠️ 警告：默认会清空results目录！
    # 如果要保留之前的结果，请使用 --no-clear 参数
    initialize_experiment_environment(
        seed=args.seed,
        clear_results=not args.no_clear,
        results_dir='results'
    )
    
    # 打印环境信息
    print_experiment_info()
    
    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    print(f"\n{'='*70}")
    print("FedDeProto: Federated Learning for Credit Risk Assessment")
    print(f"{'='*70}")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    if args.mode == 'single':
        # 运行单个实验
        print(f"Running single experiment:")
        print(f"  Dataset: {args.dataset}")
        print(f"  Partition: {args.partition_type}" + 
              (f" (α={args.alpha})" if args.partition_type == 'lda' else ""))
        print(f"  Clients: {args.num_clients}")
        print(f"  Learning Rate: {args.learning_rate}")
        print(f"  Method: {args.method}\n")
        
        metrics, history = run_single_experiment(
            dataset_name=args.dataset,
            alpha=args.alpha,
            method=args.method,
            num_clients=args.num_clients,
            learning_rate=args.learning_rate,
            partition_type=args.partition_type
        )
        
        print("\n" + "="*70)
        print("Experiment Completed!")
        print(f"Final Accuracy: {metrics['accuracy']:.4f}")
        print("="*70)
        
    elif args.mode == 'full':
        # 运行完整实验套件
        print("Running full experiment suite...")
        print("This may take several hours.\n")
        
        runner = ExperimentRunner()
        runner.run_all_experiments()
    
    elif args.mode == 'experiments':
        # 运行分组对照实验
        print("Running grouped experiments...")
        print(f"Selected groups: {args.groups}\n")
        
        from experiments.experiment_manager import ExperimentManager
        manager = ExperimentManager()
        manager.run_experiment_groups(args.groups, skip_confirm=args.yes)


if __name__ == '__main__':
    main()
