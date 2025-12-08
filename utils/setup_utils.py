"""
实用工具函数 - 随机种子设置和文件清理
"""
import os
import shutil
import random
import numpy as np
import torch
import logging
from pathlib import Path


def set_random_seed(seed=42):
    """
    设置所有随机种子以确保结果可复现
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 设置Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger = logging.getLogger('RandomSeed')
    logger.info(f"✓ Random seed set to {seed} for reproducibility")
    logger.info(f"  - Python random: {seed}")
    logger.info(f"  - NumPy: {seed}")
    logger.info(f"  - PyTorch: {seed}")
    if torch.cuda.is_available():
        logger.info(f"  - CUDA: {seed} (deterministic mode enabled)")


def clear_results_directory(results_dir='results', keep_structure=True):
    """
    清空results目录下的所有文件，但保留文件夹结构
    
    Args:
        results_dir: results目录路径
        keep_structure: 是否保留文件夹结构（默认True）
    """
    logger = logging.getLogger('FileCleaner')
    results_path = Path(results_dir)
    
    if not results_path.exists():
        logger.info(f"✓ Results directory '{results_dir}' does not exist, creating it...")
        results_path.mkdir(parents=True, exist_ok=True)
        return
    
    logger.info(f"🗑️  Clearing results directory: {results_dir}")
    
    deleted_files = 0
    deleted_dirs = 0
    
    # 遍历results目录
    for item in results_path.iterdir():
        try:
            if item.is_file():
                # 删除文件
                item.unlink()
                deleted_files += 1
                logger.debug(f"  - Deleted file: {item.name}")
            elif item.is_dir():
                if keep_structure:
                    # 保留文件夹，但清空内容
                    for sub_item in item.rglob('*'):
                        if sub_item.is_file():
                            sub_item.unlink()
                            deleted_files += 1
                            logger.debug(f"  - Deleted file: {sub_item.relative_to(results_path)}")
                else:
                    # 删除整个文件夹
                    shutil.rmtree(item)
                    deleted_dirs += 1
                    logger.debug(f"  - Deleted directory: {item.name}")
        except Exception as e:
            logger.warning(f"  ⚠️  Failed to delete {item.name}: {e}")
    
    logger.info(f"✓ Cleanup completed:")
    logger.info(f"  - Files deleted: {deleted_files}")
    if not keep_structure:
        logger.info(f"  - Directories deleted: {deleted_dirs}")
    
    # 确保必要的子目录存在
    subdirs = ['logs', 'plots', 'checkpoints', 'models']
    for subdir in subdirs:
        subdir_path = results_path / subdir
        if not subdir_path.exists():
            subdir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"  - Created directory: {subdir}")


def initialize_experiment_environment(seed=42, clear_results=True, results_dir='results'):
    """
    初始化实验环境：设置随机种子 + 清空结果目录
    
    Args:
        seed: 随机种子
        clear_results: 是否清空results目录
        results_dir: results目录路径
    """
    logger = logging.getLogger('Initialization')
    
    logger.info("="*70)
    logger.info("Initializing Experiment Environment")
    logger.info("="*70)
    
    # 1. 设置随机种子
    set_random_seed(seed)
    
    # 2. 清空results目录
    if clear_results:
        clear_results_directory(results_dir, keep_structure=True)
    
    logger.info("="*70)
    logger.info("✓ Environment initialization completed!")
    logger.info("="*70)


def get_experiment_info():
    """获取实验环境信息"""
    info = {
        'pytorch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name(0)
    
    return info


def print_experiment_info():
    """打印实验环境信息"""
    info = get_experiment_info()
    logger = logging.getLogger('ExperimentInfo')
    
    logger.info("Experiment Environment Information:")
    logger.info(f"  - PyTorch version: {info['pytorch_version']}")
    logger.info(f"  - NumPy version: {info['numpy_version']}")
    logger.info(f"  - CUDA available: {info['cuda_available']}")
    if info['cuda_available']:
        logger.info(f"  - CUDA version: {info['cuda_version']}")
        logger.info(f"  - GPU count: {info['device_count']}")
        logger.info(f"  - GPU name: {info['device_name']}")


# 测试代码
if __name__ == '__main__':
    # 配置logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(message)s'
    )
    
    print("="*70)
    print("Testing Utility Functions")
    print("="*70)
    
    # 测试1: 设置随机种子
    print("\n[Test 1] Setting random seed...")
    set_random_seed(42)
    
    # 验证随机性
    print(f"  Random number (Python): {random.random():.6f}")
    print(f"  Random number (NumPy): {np.random.rand():.6f}")
    print(f"  Random tensor (PyTorch): {torch.rand(1).item():.6f}")
    
    # 测试2: 清空results目录
    print("\n[Test 2] Clearing results directory...")
    clear_results_directory('results', keep_structure=True)
    
    # 测试3: 初始化实验环境
    print("\n[Test 3] Initializing experiment environment...")
    initialize_experiment_environment(seed=42, clear_results=True)
    
    # 测试4: 打印环境信息
    print("\n[Test 4] Printing environment information...")
    print_experiment_info()
    
    print("\n" + "="*70)
    print("✓ All tests completed!")
    print("="*70)
