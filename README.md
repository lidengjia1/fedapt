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
# 创建Python环境
conda create -n feddeproto python=3.8
conda activate feddeproto

# 安装依赖
pip install torch torchvision
pip install numpy pandas scikit-learn
pip install matplotlib seaborn tqdm
pip install openpyxl xlsxwriter  # Excel支持
```

### 2. 数据准备

将数据文件放入 `data/` 目录：
- `australian_credit.csv`
- `german_credit.csv`
- `xinwang.csv`
- `uci_credit.xls`

### 3. 快速测试

```bash
# 测试单个实验（Australian数据集 + FedAvg）
python main.py --mode single --dataset australian --method fedavg

# 查看实验组设计
python main.py --mode experiments --summary

# 运行小规模实验组（28个实验，约30分钟）
python main.py --mode experiments --groups A
```

---

## 📊 对比实验详细说明

### 实验组设计

本系统设计了 **5个实验组** 共 **228个对照实验**，用于全面评估FedDeProto性能：

| 组别 | 实验数 | 控制变量 | 研究问题 |
|------|--------|----------|----------|
| **A** | 28 | 方法对比 | 7种方法在4个数据集上的基础性能 |
| **B** | 20 | 数据划分 | 5种划分策略对FedDeProto的影响 |
| **C** | 84 | 客户端数 | 客户端数量对7种方法的影响 |
| **D** | 84 | 学习率 | 学习率对7种方法的影响 |
| **E** | 12 | 隐私预算 | 差分隐私对FedDeProto的影响 |

---

### 实验组A: 方法对比 (28个实验)

**目的**: 对比7种联邦学习方法的基础性能

**控制变量**:
- 客户端数: 10
- 学习率: 0.001
- 划分方式: LDA (α=0.1)
- 训练轮次: 150

**命令**:

```bash
# 运行实验组A所有实验
python main.py --mode experiments --groups A

# 运行单个方法在所有数据集上的实验
python main.py --mode single --method fedavg --dataset australian
python main.py --mode single --method fedavg --dataset german
python main.py --mode single --method fedavg --dataset xinwang
python main.py --mode single --method fedavg --dataset uci

# 对比FedDeProto vs FedAvg
python main.py --mode single --method feddeproto --dataset australian
python main.py --mode single --method fedavg --dataset australian
```

**7种方法**:
1. `feddeproto` - FedDeProto (本文方法)
2. `fedavg` - FedAvg (加权平均)
3. `fedprox` - FedProx (近端项正则化)
4. `fedkf` - FedKF (卡尔曼滤波)
5. `fedfa` - FedFA (特征对齐)
6. `feddr+` - FedDr+ (原型蒸馏)
7. `fedtgp` - FedTGP (时序梯度预测)

**结果文件**: `results/experiment_results_GroupA.xlsx`

---

### 实验组B: 数据划分影响 (20个实验)

**目的**: 研究不同数据异质性对FedDeProto的影响

**控制变量**:
- 方法: FedDeProto
- 客户端数: 10
- 学习率: 0.001
- 训练轮次: 150

**命令**:

```bash
# 运行实验组B所有实验
python main.py --mode experiments --groups B

# 测试不同LDA参数
python main.py --mode single --method feddeproto --dataset australian --partition-type lda --alpha 0.1
python main.py --mode single --method feddeproto --dataset australian --partition-type lda --alpha 0.3
python main.py --mode single --method feddeproto --dataset australian --partition-type lda --alpha 1.0

# 测试标签偏斜
python main.py --mode single --method feddeproto --dataset german --partition-type label_skew

# 测试特征偏斜
python main.py --mode single --method feddeproto --dataset xinwang --partition-type feature_skew
```

**5种划分策略**:
1. `lda --alpha 0.1` - 强异质性 (LDA α=0.1)
2. `lda --alpha 0.3` - 中等异质性 (LDA α=0.3)
3. `lda --alpha 1.0` - 弱异质性 (LDA α=1.0)
4. `label_skew` - 标签偏斜
5. `feature_skew` - 特征偏斜

**结果文件**: `results/experiment_results_GroupB.xlsx`

---

### 实验组C: 客户端数量影响 (84个实验)

**目的**: 研究客户端数量对所有方法的影响

**控制变量**:
- 学习率: 0.001
- 划分方式: LDA (α=0.1)
- 训练轮次: 150

**命令**:

```bash
# 运行实验组C所有实验
python main.py --mode experiments --groups C

# 测试5个客户端
python main.py --mode single --method fedavg --dataset australian --num-clients 5

# 测试10个客户端 (默认)
python main.py --mode single --method fedprox --dataset german --num-clients 10

# 测试20个客户端
python main.py --mode single --method fedkf --dataset xinwang --num-clients 20
```

**3种客户端配置**:
- 5个客户端 (小规模)
- 10个客户端 (中等规模，默认)
- 20个客户端 (大规模)

**实验矩阵**: 3种配置 × 7种方法 × 4个数据集 = 84个实验

**结果文件**: `results/experiment_results_GroupC.xlsx`

---

### 实验组D: 学习率影响 (84个实验)

**目的**: 研究学习率对所有方法的影响

**控制变量**:
- 客户端数: 10
- 划分方式: LDA (α=0.1)
- 训练轮次: 150

**命令**:

```bash
# 运行实验组D所有实验
python main.py --mode experiments --groups D

# 测试低学习率
python main.py --mode single --method fedavg --dataset australian --lr 0.0001

# 测试中等学习率 (默认)
python main.py --mode single --method fedprox --dataset german --lr 0.001

# 测试高学习率
python main.py --mode single --method fedkf --dataset xinwang --lr 0.01
```

**3种学习率**:
- 0.0001 (低学习率)
- 0.001 (中等学习率，默认)
- 0.01 (高学习率)

**实验矩阵**: 3种学习率 × 7种方法 × 4个数据集 = 84个实验

**结果文件**: `results/experiment_results_GroupD.xlsx`

---

### 实验组E: 差分隐私影响 (12个实验)

**目的**: 研究差分隐私预算对FedDeProto的影响

**控制变量**:
- 方法: FedDeProto
- 客户端数: 10
- 学习率: 0.001
- 划分方式: LDA (α=0.1)
- 训练轮次: 150

**命令**:

```bash
# 运行实验组E所有实验
python main.py --mode experiments --groups E

# 测试强隐私保护 (ε=0.5)
python main.py --mode single --method feddeproto --dataset australian --epsilon 0.5

# 测试中等隐私保护 (ε=1.0, 默认)
python main.py --mode single --method feddeproto --dataset german --epsilon 1.0

# 测试弱隐私保护 (ε=2.0)
python main.py --mode single --method feddeproto --dataset xinwang --epsilon 2.0
```

**3种隐私预算**:
- ε = 0.5 (强隐私保护)
- ε = 1.0 (中等隐私保护，默认)
- ε = 2.0 (弱隐私保护)

**实验矩阵**: 3种ε × 4个数据集 = 12个实验

**结果文件**: `results/experiment_results_GroupE.xlsx`

---

### 运行多个实验组

```bash
# 运行组A和组B (共48个实验)
python main.py --mode experiments --groups A,B

# 运行所有实验组 (共228个实验，需要数小时)
python main.py --mode experiments --groups all

# 查看实验进度和结果摘要
python main.py --mode experiments --summary
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
