# FedDeProto: 联邦学习信用风险评估系统

基于特征蒸馏和原型学习的两阶段联邦学习框架，用于信用风险评估。

## 📋 项目概述

FedDeProto 是一个创新的联邦学习框架，通过两阶段训练策略解决信用风险评估中的数据异构性和隐私保护问题。

### 核心特性

✅ **两阶段训练架构**
- 阶段1: VAE-WGAN-GP特征蒸馏 + 原型对齐 + 阈值检测
- 阶段2: 联邦分类 + 混合数据集训练

✅ **7种联邦学习方法对比**
- FedDeProto (本文方法)
- FedAvg, FedProx, FedKF, FedFA, FedDr+, FedTGP, FedFed

✅ **4个真实数据集**
- Australian Credit (692×15)
- German Credit (1002×21)
- Xinwang (17886×101)
- UCI Credit

✅ **非IID数据分区**
- Latent Dirichlet Allocation (LDA): α ∈ {0.1, 0.3, 1.0}
- Label Skew, Feature Skew, Quantity Skew

✅ **差分隐私保护**
- ε-差分隐私 (ε ∈ {0.5, 1.0, 2.0})
- Laplace/Gaussian 噪声注入

✅ **完整实验系统**
- 228个对照实验
- 5个实验组 (A-E)
- Excel自动记录结果

## 🏗️ 项目结构

```
Decision Support System/
├── config/                      # 配置文件
│   ├── base_config.py          # 基础配置
│   └── model_configs.py        # 模型架构配置
├── models/                      # 模型定义
│   ├── vae_wgan_gp.py          # VAE-WGAN-GP
│   ├── prototype.py            # 原型管理器
│   └── classifier.py           # 分类器
├── federated/                   # 联邦学习核心
│   ├── client.py               # 客户端
│   ├── server.py               # 服务器
│   └── aggregation.py          # 7种聚合策略实现
├── privacy/                     # 隐私保护
│   └── differential_privacy.py # 差分隐私
├── training/                    # 训练流程
│   ├── stage1_distillation.py  # 阶段1训练
│   └── stage2_classification.py # 阶段2训练
├── baselines/                   # 基线方法
│   └── baseline_trainer.py     # 统一训练器
├── experiments/                 # 实验管理
│   ├── experiment_logger.py    # Excel结果记录
│   ├── experiment_manager.py   # 实验调度器
│   └── run_single_dataset.py   # 单实验运行器
├── utils/                       # 工具函数
│   ├── data_loader.py          # 数据加载
│   ├── partitioner.py          # 数据分区
│   ├── metrics.py              # 评估指标
│   ├── visualization.py        # 可视化
│   └── setup_utils.py          # 环境初始化
├── data/                        # 数据目录
└── main.py                      # 主入口
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
conda create -n feddeproto python=3.8
conda activate feddeproto

# 安装依赖
pip install torch torchvision
pip install numpy pandas scikit-learn
pip install matplotlib seaborn
pip install openpyxl  # 用于读取 Excel
```

### 2. 数据准备

将以下数据文件放入 `data/` 目录：
- `australian_credit.csv`
- `german_credit.csv`
- `xinwang.csv`
- `uci_credit.xls`

### 3. 验证系统

```bash
# 测试框架核心功能
python test_framework.py

# 快速测试(运行3个样例实验)
python quick_test.py --mode quick

# 单个实验演示
python quick_test.py --mode single
```

### 4. 运行实验

#### 4.1 单个实验 (调试用)
```bash
# 基本用法
python main.py --mode single --dataset australian --method fedavg

# 完整参数
python main.py --mode single \
  --dataset australian \
  --method fedavg \
  --num-clients 10 \
  --lr 0.001 \
  --partition-type lda \
  --alpha 0.1
```

#### 4.2 分组对照实验 (推荐)
```bash
# 查看实验摘要
python experiments/experiment_manager.py --summary

# 运行特定实验组
python main.py --mode experiments --groups A      # 28个实验
python main.py --mode experiments --groups A,B    # 48个实验
python main.py --mode experiments --groups all    # 228个实验
```

**实验组说明**:
- **组A (28个)**: 基础性能对比 - 7种方法 × 4数据集
- **组B (20个)**: 数据划分影响 - 5种划分方式 × 4数据集
- **组C (84个)**: 客户端数量影响 - 3种客户端数 × 7方法 × 4数据集
- **组D (84个)**: 学习率影响 - 3种学习率 × 7方法 × 4数据集
- **组E (12个)**: 差分隐私影响 - 3种ε × 4数据集

详细说明见 **[EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)**
- `xinwang.csv`
- `uci_credit.xls`

### 3. 运行单个实验

```bash
# 使用 FedAvg 在 Australian 数据集上测试 (α=0.1)
python main.py --mode single --dataset australian --alpha 0.1 --method fedavg

# 使用 FedDeProto 在 German 数据集上测试 (α=0.3)
python main.py --mode single --dataset german --alpha 0.3 --method feddeproto

# 使用 GPU
python main.py --mode single --dataset xinwang --alpha 1.0 --method fedkf --gpu 0
```

### 4. 运行完整实验

```bash
# 运行所有数据集、所有α值、所有方法的对比实验
python main.py --mode full

# 这将运行：
# 4 datasets × 3 alpha values × 7 methods = 84 experiments
```

## 📊 实验配置

### 默认超参数

```python
# 训练轮次
T_d = 50      # 第一阶段蒸馏轮次
T_r = 100     # 第二阶段分类轮次

# 客户端配置
num_clients = 10           # 客户端数量
local_epochs = 5           # 本地训练轮次
batch_size = 64            # 批次大小

# 隐私保护
epsilon = 1.0              # 差分隐私预算
noise_type = 'laplace'     # 噪声类型

# 阈值条件
accuracy_threshold = 0.02  # 准确率波动阈值
similarity_threshold = 0.15 # 余弦相似度阈值
```

### 数据集统计

| 数据集 | 样本数 | 特征数 | 类别 |
|--------|--------|--------|------|
| Australian | 692 | 15 | 2 |
| German | 1,002 | 21 | 2 |
| Xinwang | 17,886 | 101 | 2 |
| UCI | ~30,000 | 23 | 2 |

## 📈 结果分析

实验结果保存在 `results/` 目录：

```
results/
├── logs/                        # JSON格式结果
│   ├── australian_alpha0.1_fedavg.json
│   ├── german_alpha0.3_feddeproto.json
│   └── ...
├── plots/                       # 可视化图表
│   ├── australian_alpha0.1_comparison.png
│   ├── training_curves.png
│   └── ...
└── experiment_summary_*.json    # 完整实验摘要
```

### 评估指标

- **Accuracy**: 准确率
- **Precision**: 精确率
- **Recall**: 召回率
- **F1 Score**: F1分数
- **AUC**: ROC曲线下面积

## 🔧 自定义配置

修改 `config/base_config.py` 中的参数：

```python
class BaseConfig:
    # 修改客户端数量
    num_clients = 20
    
    # 修改训练轮次
    T_d = 100
    T_r = 200
    
    # 修改隐私预算
    epsilon = 0.5
```

修改 `config/model_configs.py` 调整模型架构：

```python
'australian': {
    'encoder_hidden': [32, 16],      # 编码器层
    'decoder_hidden': [16, 32],      # 解码器层
    'classifier_hidden': [64, 32],   # 分类器层
    # ...
}
```

## 📝 方法说明

### FedDeProto (本文方法)

**第一阶段**：
- 使用 VAE-WGAN-GP 进行特征蒸馏
- 计算类原型 ω_k 并进行原型对齐
- 检测阈值条件：
  - 准确率波动 < 2%
  - 余弦相似度 < 0.15
- 生成 DP-保护的共享特征 X_s

**第二阶段**：
- 混合本地数据和共享特征
- 标准联邦分类训练
- FedAvg 聚合

### 基准方法

1. **FedAvg**: 标准联邦平均
2. **FedProx**: 添加近端项约束
3. **FedKF**: 卡尔曼滤波聚合
4. **FedFA**: 特征对齐
5. **FedDr+**: 原型驱动
6. **FedTGP**: 时间感知梯度
7. **FedFed**: 特征蒸馏

## 🔬 实验复现

完整复现论文实验：

```bash
# 1. 运行完整实验套件
python main.py --mode full

# 2. 实验将依次运行：
#    - 4个数据集 (Australian, German, Xinwang, UCI)
#    - 3个α值 (0.1, 0.3, 1.0)
#    - 7个方法 (包括FedDeProto和6个基准)

# 3. 结果将保存在 results/ 目录
#    - 对比表格
#    - 训练曲线
#    - 性能对比图
```

## 📄 引用

如果使用本代码，请引用：

```bibtex
@article{feddeproto2024,
  title={FedDeProto: Federated Learning for Credit Risk Assessment via Feature Distillation and Prototype Learning},
  author={...},
  journal={...},
  year={2024}
}
```

## 📧 联系方式

如有问题，请通过以下方式联系：
- Email: [your-email@example.com]
- GitHub Issues: [repository-link]

## 📜 许可证

MIT License

---

**注意事项**：
1. 确保数据文件正确放置在 `data/` 目录
2. 首次运行建议使用 `--mode single` 测试单个实验
3. 完整实验可能需要数小时，建议使用GPU加速
4. 定期检查 `results/` 目录保存的中间结果
