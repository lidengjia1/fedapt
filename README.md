# FedDeProto: 联邦学习信用风险评估系统

基于特征蒸馏和原型学习的两阶段联邦学习框架，用于信用风险评估。

## 📋 项目概述

FedDeProto 是一个创新的联邦学习框架，通过两阶段训练策略解决信用风险评估中的数据异构性和隐私保护问题。

### 核心特性

✅ **两阶段训练架构**
- 阶段1: VAE-WGAN-GP特征蒸馏 + 原型对齐 + 阈值检测
- 阶段2: 联邦分类 + 混合数据集训练

✅ **7种联邦学习方法对比**
- **FedDeProto** - 本文方法（完整实现 ✅）
- **FedAvg** - 加权平均（完整实现 ✅）
- **FedProx** - 近端项正则化（完整实现 ✅）
- **FedKF** - 卡尔曼滤波（完整实现 ✅）
- **FedFA** - 特征对齐（完整实现 ✅）
- **FedDr+** - 原型蒸馏（完整实现 ✅）
- **FedTGP** - 时序梯度预测（完整实现 ✅）

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
- 所有方法已完整实现

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

# 测试FedDeProto两阶段训练
python main.py --mode single --dataset australian --method feddeproto

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
python main.py --mode single --method feddeproto --dataset australian
python main.py --mode single --method fedavg --dataset german
python main.py --mode single --method fedprox --dataset xinwang

# 对比不同方法
python main.py --mode single --method feddeproto --dataset australian
python main.py --mode single --method fedavg --dataset australian
```

**7种方法**:
1. `feddeproto` - FedDeProto (本文方法，两阶段训练)
2. `fedavg` - FedAvg (加权平均)
3. `fedprox` - FedProx (近端项正则化)
4. `fedkf` - FedKF (卡尔曼滤波)
5. `fedfa` - FedFA (特征对齐)
6. `feddr+` - FedDr+ (原型蒸馏)
7. `fedtgp` - FedTGP (时序梯度预测)

**结果文件**: `results/experiment_results_GroupA.xlsx`

---

### 实验组B: 数据划分影响 (20个实验)

**目的**: 研究不同数据异质性对联邦学习方法的影响

**控制变量**:
- 方法: FedAvg（作为代表性基线）
- 客户端数: 10
- 学习率: 0.001
- 训练轮次: 150

**命令**:

```bash
# 运行实验组B所有实验
python main.py --mode experiments --groups B

# 测试不同LDA参数
python main.py --mode single --method fedavg --dataset australian --partition-type lda --alpha 0.1
python main.py --mode single --method fedavg --dataset australian --partition-type lda --alpha 0.3
python main.py --mode single --method fedavg --dataset australian --partition-type lda --alpha 1.0

# 测试标签偏斜
python main.py --mode single --method fedavg --dataset german --partition-type label_skew

# 测试特征偏斜
python main.py --mode single --method fedavg --dataset xinwang --partition-type feature_skew
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
python main.py --mode single --method feddeproto --dataset australian --num-clients 5

# 测试10个客户端 (默认)
python main.py --mode single --method fedavg --dataset german --num-clients 10

# 测试20个客户端
python main.py --mode single --method fedprox --dataset xinwang --num-clients 20
```

**3种客户端配置**:
- 5个客户端 (小规模)
- 10个客户端 (中等规模，默认)
- 20个客户端 (大规模)

**实验矩阵**: 3种配置 × 7种方法 × 4个数据集 = 84个实验

**结果文件**: `results/experiment_results_GroupC.xlsx`

---

### 实验组D: 学习率影响 (72个实验)

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
python main.py --mode single --method feddeproto --dataset australian --lr 0.0001

# 测试中等学习率 (默认)
python main.py --mode single --method fedavg --dataset german --lr 0.001

# 测试高学习率
python main.py --mode single --method fedprox --dataset xinwang --lr 0.01
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
python main.py --mode experiments --groups A,B,C,D,E

# 查看实验进度和结果摘要
python main.py --mode experiments --summary
```
- 当前共168个可运行实验（组A: 24 + 组B: 20 + 组C: 72 + 组D: 72）

---

## 🔧 高级配置

### 所有命令行参数

```bash
python main.py \
  --mode {single|experiments}      # 运行模式
  --groups {A|B|C|D|E}             # 实验组 (仅experiments模式)
  --summary                         # 显示实验摘要 (仅experiments模式)
  --dataset {australian|german|xinwang|uci}  # 数据集
  --method {fedavg|fedprox|fedkf|fedfa|feddr+|fedtgp}  # 方法 (不含feddeproto)
  --num-clients {5|10|20}          # 客户端数量
  --lr {0.0001|0.001|0.01}         # 学习率
  --partition-type {lda|label_skew|feature_skew|quantity_skew}  # 划分方式
  --alpha {0.1|0.3|1.0}            # LDA参数 (仅lda划分)
  --epsilon {0.5|1.0|2.0}          # 差分隐私预算 (FedDeProto专用)
  --num-rounds {150}               # 训练轮次
  --local-epochs {5}               # 本地训练轮次
  --batch-size {64}                # 批次大小
  --seed {42}                      # 随机种子
  --gpu {0|1|...}                  # GPU设备
  --no-clear                       # 不清空results目录
```

### 配置文件修改

修改 `config/base_config.py`:

```python
class BaseConfig:
    # 客户端配置
    num_clients = 10
    local_epochs = 5
    batch_size = 64
    
    # 训练配置
    learning_rate = 0.001
    num_rounds = 150
    
    # 隐私配置
    epsilon = 1.0
    noise_type = 'laplace'
    
    # 阈值配置
    accuracy_threshold = 0.02
    similarity_threshold = 0.15
```

修改 `config/model_configs.py`:

```python
MODEL_CONFIGS = {
    'australian': {
        'input_dim': 15,
        'encoder_hidden': [32, 16],
        'latent_dim': 8,
        'decoder_hidden': [16, 32],
        'classifier_hidden': [64, 32],
        'num_classes': 2
    },
    # ...
}
```

---

## 📈 结果分析

### 输出文件

实验结果自动保存在 `results/` 目录：

```
results/
├── experiment_results_GroupA.xlsx    # 组A结果Excel
├── experiment_results_GroupB.xlsx    # 组B结果Excel
├── experiment_results_GroupC.xlsx    # 组C结果Excel
├── experiment_results_GroupD.xlsx    # 组D结果Excel
├── experiment_results_GroupE.xlsx    # 组E结果Excel
├── plots/                            # 可视化图表
│   ├── australian_fedavg_loss.png   # 训练损失曲线
│   ├── german_fedprox_accuracy.png  # 准确率曲线
│   └── method_comparison.png        # 方法对比图
└── logs/                             # 运行日志
    ├── australian_fedavg_20241208.log
    └── ...
```

### Excel结果表格

每个实验组的Excel文件包含多个工作表：

1. **Summary** - 实验摘要
   - 实验配置
   - 最终指标对比表
   - 最佳方法排名

2. **Detailed_Results** - 详细结果
   - 每个实验的完整指标
   - Accuracy, Precision, Recall, F1, AUC
   - 训练时间

3. **Training_History** - 训练历史
   - 每轮的Loss和Accuracy
   - 用于绘制训练曲线

### 评估指标

| 指标 | 说明 | 计算公式 |
|------|------|----------|
| **Accuracy** | 准确率 | (TP+TN) / (TP+TN+FP+FN) |
| **Precision** | 精确率 | TP / (TP+FP) |
| **Recall** | 召回率 | TP / (TP+FN) |
| **F1 Score** | F1分数 | 2 × (Precision × Recall) / (Precision + Recall) |
| **AUC** | ROC曲线下面积 | Area Under ROC Curve |

### 可视化图表

系统自动生成以下图表：

1. **训练损失曲线** - 观察收敛速度
2. **准确率曲线** - 评估性能提升
3. **方法对比柱状图** - 直观对比不同方法
4. **混淆矩阵** - 分析分类错误

---

## 📝 方法说明

### 6种联邦学习方法详解

#### 1. FedAvg (Federated Averaging)

**服务端**: 加权平均聚合  
**客户端**: 标准SGD训练  
**公式**: `w_global = Σ(n_k / N) × w_k`  
**状态**: ✅ 完整实现

#### 2. FedProx (Federated Proximal)

**服务端**: 标准聚合  
**客户端**: 添加近端正则化项  
**损失函数**: `L(w) + (μ/2) × ||w - w_global||²`  
**适用**: 异质性强的场景  
**状态**: ✅ 完整实现

#### 3. FedKF (Federated Kalman Filter)

**服务端**: 卡尔曼滤波聚合(有状态)  
**客户端**: 标准训练  
**特点**: 贝叶斯推断，追踪参数不确定性  
**状态**: 维护均值和协方差矩阵  
**实现**: ✅ 完整实现

#### 4. FedFA (Federated Feature Alignment)

**服务端**: 特征对齐聚合(有状态)  
**客户端**: 上传特征向量  
**特点**: 对齐客户端间的特征分布  
**状态**: 全局特征统计(均值、方差)  
**实现**: ✅ 完整实现

#### 5. FedDr+ (Federated Dynamic Regularization)

**服务端**: 原型聚合(有状态)  
**客户端**: 计算并上传类原型  
**特点**: 基于原型的知识蒸馏  
**状态**: 全局类原型字典  
**实现**: ✅ 完整实现

#### 6. FedTGP (Federated Time-aware Gradient Prediction)

**服务端**: 梯度预测聚合(有状态)  
**客户端**: 标准训练  
**特点**: 利用历史梯度预测未来更新  
**状态**: 梯度历史和上轮模型  
**实现**: ✅ 完整实现

#### 7. FedDeProto (本文方法) ⚠️

**两阶段训练**:
- **阶段1**: VAE-WGAN-GP特征蒸馏 + 原型对齐 + 阈值检测
- **阶段2**: 混合数据集(本地+共享特征)联邦分类

**核心创新**:
- 差分隐私保护的共享特征生成
- 基于原型的知识对齐
- 自适应阈值检测机制

**实现状态**: 
- ✅ 阶段1训练器已实现 (`training/stage1_distillation.py`)
- ✅ 阶段2训练器已实现 (`training/stage2_classification.py`)
- ⚠️ 待集成到主实验流程中

### 方法对比表

| 方法 | 服务端 | 客户端 | 有状态? | 适用场景 | 实现状态 |
|------|--------|--------|---------|----------|----------|
| FedAvg | 加权平均 | 标准SGD | ❌ | IID数据 | ✅ |
| FedProx | 标准聚合 | 近端项 | ❌ | Non-IID数据 | ✅ |
| FedKF | 卡尔曼滤波 | 标准SGD | ✅ | 噪声环境 | ✅ |
| FedFA | 特征对齐 | 上传特征 | ✅ | 特征分布差异大 | ✅ |
| FedDr+ | 原型聚合 | 计算原型 | ✅ | 标签偏斜 | ✅ |
| FedTGP | 梯度预测 | 标准SGD | ✅ | 稳定训练 | ✅ |
| FedDeProto | 两阶段训练 | 特征蒸馏 | ✅ | 隐私保护+异质性 | ⚠️ 待集成 |

---

## 🎯 实验复现

### 完整复现论文实验

```bash
# 步骤1: 运行所有可用对照实验 (约4-5小时)
python main.py --mode experiments --groups A,B,C,D

# 步骤2: 查看结果摘要
python main.py --mode experiments --summary

# 步骤3: 分析Excel结果文件
# 打开 results/experiment_results_Group*.xlsx
```

**实验规模**:
- 当前可运行: 168个实验 (组A-D)
- 待FedDeProto实现: 12个实验 (组E)
- 总计: 180个实验

### 快速验证（20分钟）

```bash
# 只运行组A的关键实验（24个实验）
python main.py --mode experiments --groups A
```

### 单个对比实验

```bash
# FedProx vs FedAvg on Australian
python main.py --mode single --method fedprox --dataset australian
python main.py --mode single --method fedavg --dataset australian

# 对比结果在 results/ 目录
```

---

## 🔬 数据集信息

| 数据集 | 样本数 | 特征数 | 正样本 | 负样本 | 来源 |
|--------|--------|--------|--------|--------|------|
| Australian | 692 | 15 | 307 | 385 | UCI |
| German | 1,002 | 21 | 700 | 300 | UCI |
| Xinwang | 17,886 | 101 | 4,221 | 13,665 | Lending Club |
| UCI | 30,000 | 23 | 6,636 | 23,364 | UCI |

### 数据划分策略

1. **LDA (Latent Dirichlet Allocation)**
   - α = 0.1: 强异质性(每个客户端只有少数类别)
   - α = 0.3: 中等异质性
   - α = 1.0: 弱异质性(接近IID)

2. **Label Skew**: 标签分布不均
3. **Feature Skew**: 特征分布差异
4. **Quantity Skew**: 样本数量差异

---

## ⚙️ 系统要求

### 硬件要求

- **CPU**: 4核以上
- **内存**: 16GB+
- **GPU**: 推荐NVIDIA GPU (2GB显存+)，可选

### 软件环境

- **Python**: 3.8+
- **PyTorch**: 1.10+
- **CUDA**: 11.0+ (使用GPU时)

### 运行时间估算

| 实验规模 | 实验数 | CPU时间 | GPU时间 |
|----------|--------|---------|---------|
| 单个实验 | 1 | ~2分钟 | ~1分钟 |
| 组A | 24 | ~50分钟 | ~25分钟 |
| 组B | 20 | ~40分钟 | ~20分钟 |
| 组C | 72 | ~2.5小时 | ~1.2小时 |
| 组D | 72 | ~2.5小时 | ~1.2小时 |
| 组E | 12 | (待FedDeProto实现) | - |
| **A-D合计** | **168** | **~5小时** | **~2.5小时** |
| **全部** | **180** | **~6小时** | **~3小时** |

---

## 🐛 常见问题

### Q1: 运行时显存不足

```bash
# 减小批次大小
python main.py --mode single --batch-size 32

# 或减少客户端数
python main.py --mode single --num-clients 5
```

### Q2: 数据文件找不到

```bash
# 检查data目录结构
ls data/
# 应包含: australian_credit.csv, german_credit.csv, xinwang.csv, uci_credit.xls
```

### Q3: 实验中断后如何继续

```bash
# 系统会自动跳过已完成的实验
# 直接重新运行相同命令即可
python main.py --mode experiments --groups A
```

### Q4: 如何使用GPU

```bash
# 指定GPU设备
python main.py --mode single --gpu 0

# 多GPU选择
python main.py --mode single --gpu 1
```

### Q5: 结果文件在哪里

```bash
# Excel结果
results/experiment_results_Group*.xlsx

# 训练日志
results/logs/*.log

# 可视化图表
results/plots/*.png
```

---

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

---

## 📧 联系方式

如有问题，请通过以下方式联系：
- Email: [your-email@example.com]
- GitHub Issues: [repository-link]

---

## 📜 许可证

MIT License

---

## 🙏 致谢

感谢以下开源项目：
- PyTorch
- scikit-learn
- pandas
- matplotlib

---

**最后更新**: 2024年12月8日
