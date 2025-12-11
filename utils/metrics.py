"""
评估指标模块
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch


def evaluate_model(model, test_loader, device='cpu'):
    """
    评估模型性能
    
    Args:
        model: 待评估模型
        test_loader: 测试数据加载器
        device: 设备
    
    Returns:
        metrics: 评估指标字典
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            
            # 获取预测
            logits = model(batch_X)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算指标（二分类，pos_label=1, zero_division=0）
    metrics = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions, average='binary', pos_label=1, zero_division=0),
        'recall': recall_score(all_labels, all_predictions, average='binary', pos_label=1, zero_division=0),
        'f1_score': f1_score(all_labels, all_predictions, average='binary', pos_label=1, zero_division=0),
        'confusion_matrix': confusion_matrix(all_labels, all_predictions)
    }
    
    # 计算 AUC (仅对二分类)
    if all_probs.shape[1] == 2:
        try:
            metrics['auc'] = roc_auc_score(all_labels, all_probs[:, 1])
        except ValueError:
            metrics['auc'] = 0.0  # 如果只有一个类别，AUC无法计算
    
    return metrics


def print_metrics(metrics, method_name="Model"):
    """打印评估指标"""
    print(f"\n{'='*50}")
    print(f"{method_name} Evaluation Results")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    if 'auc' in metrics:
        print(f"AUC:       {metrics['auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print(f"{'='*50}\n")


def compare_methods(results_dict):
    """
    比较多个方法的性能
    
    Args:
        results_dict: {method_name: metrics_dict}
    """
    print(f"\n{'='*80}")
    print("Method Comparison")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
    print(f"{'-'*80}")
    
    for method_name, metrics in results_dict.items():
        auc_str = f"{metrics.get('auc', 0):.4f}" if 'auc' in metrics else "N/A"
        print(f"{method_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f} {auc_str:<12}")
    
    print(f"{'='*80}\n")
