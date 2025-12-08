"""
可视化模块
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_training_curves(history, save_path=None):
    """
    绘制训练曲线
    
    Args:
        history: 训练历史，格式：{'loss': [...], 'accuracy': [...]}
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    if 'loss' in history:
        axes[0].plot(history['loss'], marker='o', linewidth=2)
        axes[0].set_xlabel('Round', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    if 'accuracy' in history:
        axes[1].plot(history['accuracy'], marker='s', color='green', linewidth=2)
        axes[1].set_xlabel('Round', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training Accuracy', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(cm, class_names=['Negative', 'Positive'], save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称
        save_path: 保存路径
    """
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_method_comparison(results_dict, metric='accuracy', save_path=None):
    """
    绘制方法比较图
    
    Args:
        results_dict: {method_name: metrics_dict}
        metric: 要比较的指标
        save_path: 保存路径
    """
    methods = list(results_dict.keys())
    values = [results_dict[m][metric] for m in methods]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(methods)), values, color='steelblue', alpha=0.8)
    
    # 在柱状图上添加数值
    for i, (bar, val) in enumerate(zip(bars, values)):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.title(f'{metric.capitalize()} Comparison Across Methods', 
             fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_privacy_accuracy_tradeoff(epsilon_values, accuracy_values, save_path=None):
    """
    绘制隐私-准确率权衡曲线
    
    Args:
        epsilon_values: ε值列表
        accuracy_values: 对应的准确率列表
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(epsilon_values, accuracy_values, marker='o', 
            linewidth=2, markersize=8, color='darkred')
    
    plt.xlabel('Privacy Budget (ε)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Privacy-Accuracy Tradeoff', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 添加数值标注
    for eps, acc in zip(epsilon_values, accuracy_values):
        plt.annotate(f'{acc:.3f}', (eps, acc), 
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Privacy-accuracy tradeoff plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_alpha_comparison(results_by_alpha, metric='accuracy', save_path=None):
    """
    比较不同α值下的性能
    
    Args:
        results_by_alpha: {alpha: {method: metrics}}
        metric: 要比较的指标
        save_path: 保存路径
    """
    alphas = sorted(results_by_alpha.keys())
    methods = list(next(iter(results_by_alpha.values())).keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(alphas))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        values = [results_by_alpha[alpha][method][metric] for alpha in alphas]
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=method, alpha=0.8)
    
    ax.set_xlabel('Alpha (α)', fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f'{metric.capitalize()} Comparison Across Different α Values', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'α={a}' for a in alphas])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Alpha comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_results_directory(base_path='results'):
    """创建结果目录"""
    results_dir = Path(base_path)
    results_dir.mkdir(exist_ok=True)
    
    subdirs = ['plots', 'logs', 'models']
    for subdir in subdirs:
        (results_dir / subdir).mkdir(exist_ok=True)
    
    return results_dir
