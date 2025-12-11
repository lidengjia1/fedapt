"""
实验管理器 - 管理分组对照实验
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from datetime import datetime
import time

from config.base_config import BaseConfig
from utils.results_logger import ExperimentLogger, ExperimentProgressTracker
from experiments.run_single_dataset import run_single_experiment
from utils.setup_utils import set_random_seed


class ExperimentManager:
    """分组对照实验管理器"""
    
    def __init__(self):
        self.config = BaseConfig()
        self.logger = ExperimentLogger('results/experiment_results.xlsx')
        
        # 实验配置
        self.datasets = ['australian', 'german', 'xinwang', 'uci']
        self.methods = ['feddeproto', 'fedavg', 'fedprox', 'fedkf', 'fedfa', 'feddr+', 'fedtgp']
        
        # 参数设置
        self.learning_rates = [0.01]  # 统一使用0.01
        self.client_numbers = [5, 8, 10]  # 修改为5,8,10
        self.alphas = [0.1, 0.3, 1.0]
        self.epsilons = [0.1, 1.0, 10.0]
        self.partition_types = ['lda', 'quantity_skew']  # 删除label_skew和feature_skew
        
        # 实验组定义
        self.experiment_groups = {
            'A': self._define_group_A,
            'B': self._define_group_B,
            'C': self._define_group_C,
            'D': self._define_group_D
        }
        
        print("="*70)
        print("FedDeProto 分组对照实验系统")
        print("="*70)
        print(f"数据集: {self.datasets}")
        print(f"方法: {self.methods}")
        print(f"学习率: {self.learning_rates}")
        print(f"客户端数: {self.client_numbers}")
        print(f"Alpha值: {self.alphas}")
        print(f"隐私预算: {self.epsilons}")
        print("="*70)
    
    def _define_group_A(self):
        """
        实验组A: 基础性能对比
        固定: 10客户端, LDA α=0.1, lr=0.01, ε=1.0
        变量: 4数据集 × 7方法 = 28个实验
        """
        experiments = []
        for dataset in self.datasets:
            for method in self.methods:
                exp = {
                    'group': 'A',
                    'dataset': dataset,
                    'partition_type': 'lda',
                    'alpha': 0.1,
                    'num_clients': 10,
                    'learning_rate': 0.01,  # 修改为0.01
                    'epsilon': 1.0,
                    'method': method,
                    'description': '基础性能对比'
                }
                experiments.append(exp)
        return experiments
    
    def _define_group_B(self):
        """
        实验组B: 数据划分方式影响
        固定: 10客户端, lr=0.01, ε=1.0, 7方法
        变量: 4数据集 × (3LDA_α + 1QuantitySkew) × 7方法 = 112个实验
        """
        experiments = []
        for dataset in self.datasets:
            # LDA with different alpha
            for alpha in self.alphas:
                for method in self.methods:
                    exp = {
                        'group': 'B',
                        'dataset': dataset,
                        'partition_type': 'lda',
                        'alpha': alpha,
                        'num_clients': 10,
                        'learning_rate': 0.01,
                        'epsilon': 1.0,
                        'method': method,
                        'description': f'LDA划分 α={alpha}'
                    }
                    experiments.append(exp)
            
            # Quantity Skew
            for method in self.methods:
                exp = {
                    'group': 'B',
                    'dataset': dataset,
                    'partition_type': 'quantity_skew',
                    'alpha': None,
                    'num_clients': 10,
                    'learning_rate': 0.01,
                    'epsilon': 1.0,
                    'method': method,
                    'description': '数量偏斜划分'
                }
                experiments.append(exp)
        
        return experiments
    
    def _define_group_C(self):
        """
        实验组C: 客户端数量影响
        固定: LDA α=0.1, lr=0.01, ε=1.0
        变量: 4数据集 × 3客户端数(5,8,10) × 7方法 = 84个实验
        """
        experiments = []
        for dataset in self.datasets:
            for num_clients in self.client_numbers:
                for method in self.methods:
                    exp = {
                        'group': 'C',
                        'dataset': dataset,
                        'partition_type': 'lda',
                        'alpha': 0.1,
                        'num_clients': num_clients,
                        'learning_rate': 0.01,
                        'epsilon': 1.0,
                        'method': method,
                        'description': f'{num_clients}个客户端'
                    }
                    experiments.append(exp)
        return experiments
    
    def _define_group_D(self):
        """
        实验组D: 差分隐私影响 (仅FedDeProto)
        固定: 10客户端, LDA α=0.1, lr=0.01
        变量: 4数据集 × 3ε = 12个实验
        """
        experiments = []
        for dataset in self.datasets:
            for epsilon in self.epsilons:
                exp = {
                    'group': 'E',
                    'dataset': dataset,
                    'partition_type': 'lda',
                    'alpha': 0.1,
                    'num_clients': 10,
                    'learning_rate': 0.01,
                    'epsilon': epsilon,
                    'method': 'feddeproto',
                    'description': f'隐私预算ε={epsilon}'
                }
                experiments.append(exp)
        return experiments
    
    def run_experiment_groups(self, groups='all', skip_confirm=False):
        """
        运行实验组
        
        Args:
            groups: 要运行的实验组 ('all', 'A', 'B,C', etc.)
            skip_confirm: 是否跳过确认提示，直接运行
        """
        # 解析要运行的实验组
        if groups == 'all':
            selected_groups = list(self.experiment_groups.keys())
        else:
            selected_groups = [g.strip().upper() for g in groups.split(',')]
        
        # 收集所有实验
        all_experiments = []
        for group in selected_groups:
            if group in self.experiment_groups:
                exps = self.experiment_groups[group]()
                all_experiments.extend(exps)
                print(f"✓ 实验组 {group}: {len(exps)} 个实验")
        
        total_count = len(all_experiments)
        print(f"\n总计: {total_count} 个实验")
        print("="*70)
        
        # 确认是否继续
        if not skip_confirm:
            response = input(f"\n是否开始运行 {total_count} 个实验? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("已取消")
                return
        else:
            print(f"\n自动开始运行 {total_count} 个实验...")
        
        # 创建进度追踪器
        tracker = ExperimentProgressTracker(total_count)
        
        # 运行实验
        for i, exp_config in enumerate(all_experiments, 1):
            print(f"\n\n{'#'*70}")
            print(f"实验 {i}/{total_count}")
            print(f"实验组: {exp_config['group']} - {exp_config['description']}")
            print(f"配置: {exp_config['dataset']}, {exp_config['method']}, "
                  f"clients={exp_config['num_clients']}, lr={exp_config['learning_rate']}")
            print(f"{'#'*70}\n")
            
            try:
                # 运行单个实验
                result = self._run_single_experiment(exp_config)
                
                # 记录结果
                self.logger.log_result(exp_config['group'], result)
                
                # 更新进度
                tracker.update(result['Experiment_ID'])
                
                # 每10个实验保存一次
                if i % 10 == 0:
                    self.logger.save_to_excel()
                    print(f"✓ 已保存中间结果 ({i}/{total_count})")
                
            except Exception as e:
                print(f"❌ 实验失败: {e}")
                # 记录失败的实验
                error_result = self._create_error_result(exp_config, str(e))
                self.logger.log_result(exp_config['group'], error_result)
                continue
        
        # 保存最终结果
        print("\n" + "="*70)
        print("所有实验完成！正在保存结果...")
        print("="*70)
        self.logger.save_to_excel()
        self.logger.export_json()
        
        print(f"\n✓ 实验完成！共完成 {total_count} 个实验")
        print(f"✓ 结果已保存到: {self.logger.excel_path}")
    
    def _run_single_experiment(self, exp_config):
        """运行单个实验"""
        start_time = time.time()
        
        # 运行实验
        metrics, history = run_single_experiment(
            dataset_name=exp_config['dataset'],
            alpha=exp_config.get('alpha', 0.1),
            method=exp_config['method'],
            num_clients=exp_config['num_clients'],
            learning_rate=float(exp_config['learning_rate']),
            partition_type=exp_config['partition_type']
        )
        
        training_time = time.time() - start_time
        
        # 提取训练信息
        training_info = {
            'total_rounds': len(history.get('test_accuracy', [])),
            'convergence_round': self._find_convergence_round(history),
            'training_time': training_time,
            'avg_round_time': training_time / max(len(history.get('test_accuracy', [])), 1),
            'final_loss': self._get_final_loss(history, exp_config['method']),
            'gpu_used': 'Yes' if torch.cuda.is_available() else 'No'
        }
        
        # 如果是FedDeProto，添加额外信息
        if exp_config['method'] == 'feddeproto':
            training_info.update({
                'stage1_rounds': history.get('stage1_rounds', '-'),
                'stage2_rounds': history.get('stage2_rounds', '-'),
                'stopped_clients': history.get('stopped_clients', '-')
            })
        
        # 创建结果字典
        result = self.logger.create_result_dict(
            dataset=exp_config['dataset'],
            partition_type=exp_config['partition_type'],
            alpha=exp_config.get('alpha'),
            num_clients=exp_config['num_clients'],
            learning_rate=float(exp_config['learning_rate']),
            epsilon=exp_config['epsilon'],
            method=exp_config['method'],
            metrics=metrics,
            training_info=training_info,
            notes=exp_config['description']
        )
        
        return result
    
    def _find_convergence_round(self, history):
        """找到收敛轮次（准确率不再显著提升）"""
        if 'test_accuracy' not in history or not history['test_accuracy']:
            return 0
        
        accuracies = history['test_accuracy']
        threshold = 0.001  # 0.1% 的提升阈值
        
        for i in range(10, len(accuracies)):
            # 检查最近10轮的平均提升
            recent_improvement = np.mean(np.diff(accuracies[i-10:i]))
            if recent_improvement < threshold:
                return i
        
        return len(accuracies)
    
    def _create_error_result(self, exp_config, error_msg):
        """创建错误结果记录"""
        return self.logger.create_result_dict(
            dataset=exp_config['dataset'],
            partition_type=exp_config['partition_type'],
            alpha=exp_config.get('alpha'),
            num_clients=exp_config['num_clients'],
            learning_rate=exp_config['learning_rate'],
            epsilon=exp_config['epsilon'],
            method=exp_config['method'],
            metrics={'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0},
            training_info={'total_rounds': 0, 'convergence_round': 0, 
                          'training_time': 0, 'avg_round_time': 0, 
                          'final_loss': 0, 'gpu_used': 'Error'},
            notes=f'ERROR: {error_msg}'
        )
    
    def _get_final_loss(self, history, method):
        """
        获取最终loss值
        对于FedDeProto，只取stage2的最后一个loss（分类loss）
        对于其他方法，取train_loss的最后一个值
        """
        if method == 'feddeproto':
            # FedDeProto: 只取stage2的loss
            stage2_rounds = history.get('stage2_rounds', 0)
            if stage2_rounds > 0:
                train_loss = history.get('train_loss', [])
                if len(train_loss) >= stage2_rounds:
                    # 取最后stage2_rounds个loss的最后一个
                    stage2_loss = train_loss[-stage2_rounds:]
                    return stage2_loss[-1] if stage2_loss else 0
        
        # 其他方法：取最后一个loss
        train_loss = history.get('train_loss', [0])
        return train_loss[-1] if train_loss else 0
    
    def print_experiment_summary(self):
        """打印实验配置摘要"""
        print("\n" + "="*70)
        print("实验配置摘要")
        print("="*70)
        
        for group_name, group_func in self.experiment_groups.items():
            experiments = group_func()
            print(f"\n实验组 {group_name}: {len(experiments)} 个实验")
            print(f"描述: {experiments[0]['description'] if experiments else 'N/A'}")
        
        total = sum(len(func()) for func in self.experiment_groups.values())
        print(f"\n总计: {total} 个实验")
        print("="*70)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FedDeProto 分组对照实验系统')
    parser.add_argument('--groups', type=str, default='all',
                       help='要运行的实验组 (all, A, B, C, D, E, 或组合如 A,B,C)')
    parser.add_argument('--summary', action='store_true',
                       help='只显示实验摘要,不运行实验')
    
    args = parser.parse_args()
    
    manager = ExperimentManager()
    
    if args.summary:
        manager.print_experiment_summary()
    else:
        manager.run_experiment_groups(args.groups)


if __name__ == '__main__':
    main()
