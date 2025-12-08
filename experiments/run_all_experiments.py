"""
完整实验套件 - 运行所有数据集、所有α值、所有方法的对比实验
"""
import torch
import numpy as np
import json
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from config.base_config import BaseConfig
from config.model_configs import get_model_config
from utils.data_loader import CreditDataLoader
from utils.partitioner import DataPartitioner
from models.classifier import create_classifier
from baselines.baseline_trainer import BaselineTrainer
from training.stage1_distillation import Stage1Distillation
from training.stage2_classification import Stage2Classification
from utils.metrics import evaluate_model, compare_methods
from utils.visualization import (
    plot_method_comparison, plot_alpha_comparison,
    create_results_directory
)
from torch.utils.data import TensorDataset, DataLoader, Subset


class ExperimentRunner:
    """完整实验运行器"""
    
    def __init__(self):
        self.datasets = ['australian', 'german', 'xinwang', 'uci']
        self.alphas = [0.1, 0.3, 1.0]
        self.methods = ['feddeproto', 'fedavg', 'fedkf', 'fedfa', 'feddr+', 'fedtgp', 'fedfed']
        
        self.config = BaseConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config.device = self.device
        
        self.results_dir = create_results_directory('results')
        self.all_results = {}
        
        print(f"Experiment Runner Initialized")
        print(f"Device: {self.device}")
        print(f"Datasets: {self.datasets}")
        print(f"Alpha values: {self.alphas}")
        print(f"Methods: {self.methods}")
    
    def prepare_data(self, dataset_name, alpha):
        """准备数据和客户端加载器"""
        # 加载数据
        data_loader = CreditDataLoader(data_dir='data')
        X_train, X_test, y_train, y_test = data_loader.load_dataset(dataset_name)
        
        # 数据分区
        partitioner = DataPartitioner(self.config.num_clients)
        client_indices = partitioner.partition_lda(y_train, alpha=alpha)
        
        # 创建数据加载器
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
                batch_size=self.config.batch_size,
                shuffle=True
            )
            for indices in client_indices
        ]
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        return client_loaders, test_loader, X_train.shape[1]
    
    def run_feddeproto(self, client_loaders, test_loader, config):
        """运行FedDeProto方法"""
        # Stage 1
        stage1 = Stage1Distillation(
            client_loaders=client_loaders,
            test_loader=test_loader,
            config=config
        )
        shared_dataset = stage1.train()
        
        # Stage 2
        stage2 = Stage2Classification(
            client_loaders=client_loaders,
            shared_dataset=shared_dataset,
            test_loader=test_loader,
            config=config
        )
        history = stage2.train()
        
        return stage2.global_model, history
    
    def run_baseline(self, method, model, client_loaders, test_loader, config):
        """运行基准方法"""
        trainer = BaselineTrainer(
            model=model,
            client_data_loaders=client_loaders,
            test_loader=test_loader,
            config=config,
            method=method
        )
        
        num_rounds = config.T_d + config.T_r
        history = trainer.train(num_rounds)
        
        return trainer.global_model, history
    
    def run_single_experiment(self, dataset_name, alpha, method):
        """运行单个实验配置"""
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name.upper()} | Alpha: {alpha} | Method: {method.upper()}")
        print(f"{'='*70}")
        
        try:
            # 准备数据
            client_loaders, test_loader, input_dim = self.prepare_data(dataset_name, alpha)
            
            # 获取模型配置
            model_config = get_model_config(dataset_name)
            
            # 创建模型
            model = create_classifier(
                classifier_type=model_config['classifier_type'],
                input_dim=input_dim,
                hidden_dims=model_config['classifier_hidden']
            )
            model.to(self.device)
            
            # 训练
            if method == 'feddeproto':
                final_model, history = self.run_feddeproto(
                    client_loaders, test_loader, self.config
                )
            else:
                final_model, history = self.run_baseline(
                    method, model, client_loaders, test_loader, self.config
                )
            
            # 评估
            metrics = evaluate_model(final_model, test_loader, self.device)
            
            result = {
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
                'final_accuracy': float(metrics['accuracy'])
            }
            
            print(f"✓ Completed - Accuracy: {metrics['accuracy']:.4f}")
            
            return result
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            return None
    
    def run_all_experiments(self):
        """运行所有实验"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n{'#'*80}")
        print(f"# Starting Full Experiment Suite - {timestamp}")
        print(f"# Total experiments: {len(self.datasets) * len(self.alphas) * len(self.methods)}")
        print(f"{'#'*80}\n")
        
        experiment_count = 0
        
        for dataset in self.datasets:
            self.all_results[dataset] = {}
            
            for alpha in self.alphas:
                self.all_results[dataset][alpha] = {}
                
                for method in self.methods:
                    experiment_count += 1
                    print(f"\n[{experiment_count}/{len(self.datasets)*len(self.alphas)*len(self.methods)}] ", end='')
                    
                    result = self.run_single_experiment(dataset, alpha, method)
                    
                    if result:
                        self.all_results[dataset][alpha][method] = result
                        
                        # 保存单个结果
                        result_file = self.results_dir / 'logs' / f'{dataset}_alpha{alpha}_{method}.json'
                        with open(result_file, 'w') as f:
                            json.dump(result, f, indent=4)
        
        # 保存完整结果
        self.save_all_results(timestamp)
        
        # 生成对比报告
        self.generate_comparison_report()
        
        print(f"\n{'#'*80}")
        print(f"# All experiments completed!")
        print(f"# Results saved to: {self.results_dir}")
        print(f"{'#'*80}\n")
    
    def save_all_results(self, timestamp):
        """保存所有结果"""
        summary_file = self.results_dir / f'experiment_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(self.all_results, f, indent=4)
        print(f"\nFull results saved to {summary_file}")
    
    def generate_comparison_report(self):
        """生成对比报告和可视化"""
        print("\n" + "="*70)
        print("Generating Comparison Report")
        print("="*70)
        
        for dataset in self.datasets:
            print(f"\n### Dataset: {dataset.upper()} ###")
            
            for alpha in self.alphas:
                print(f"\n--- Alpha = {alpha} ---")
                
                if dataset in self.all_results and alpha in self.all_results[dataset]:
                    results = self.all_results[dataset][alpha]
                    
                    # 打印对比表格
                    method_metrics = {
                        method: data['metrics']
                        for method, data in results.items()
                    }
                    
                    if method_metrics:
                        compare_methods(method_metrics)
                        
                        # 绘制对比图
                        plot_path = self.results_dir / 'plots' / f'{dataset}_alpha{alpha}_comparison.png'
                        plot_method_comparison(
                            method_metrics,
                            metric='accuracy',
                            save_path=plot_path
                        )


def main():
    """主函数"""
    runner = ExperimentRunner()
    runner.run_all_experiments()


if __name__ == '__main__':
    main()
