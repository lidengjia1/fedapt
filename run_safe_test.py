"""
安全测试脚本 - 使用更保守的参数避免NaN问题
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from experiments.run_single_dataset import run_single_experiment
from utils.setup_utils import initialize_experiment_environment

# 初始化环境
initialize_experiment_environment(seed=42, clear_results=False)

print("="*70)
print("Running SAFE test with conservative parameters")
print("="*70)
print("Configuration:")
print("  - Learning Rate: 0.005 (reduced from 0.02)")
print("  - Clients: 10 (more balanced)")
print("  - Alpha: 0.3 (less extreme partitioning)")
print("="*70 + "\n")

# 运行测试
try:
    metrics, history = run_single_experiment(
        dataset_name='australian',
        alpha=0.3,  # 更温和的数据划分
        method='feddeproto',
        num_clients=10,  # 更多客户端，减少单类别概率
        learning_rate=0.005,  # 降低学习率
        partition_type='lda'
    )
    
    print("\n" + "="*70)
    print("✓ Test Completed Successfully!")
    print(f"Final Accuracy: {metrics['accuracy']:.4f}")
    print("="*70)
    
except Exception as e:
    print("\n" + "="*70)
    print("❌ Test Failed!")
    print(f"Error: {e}")
    print("="*70)
    import traceback
    traceback.print_exc()
