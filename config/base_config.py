"""
FedDeProto - 基础配置文件
定义所有超参数和训练配置
"""

class BaseConfig:
    """基础配置类"""
    
    def __init__(self):
        """初始化配置，将类变量转换为实例属性（小写形式）"""
        # 数据集配置
        self.datasets = self.DATASETS
        self.data_dir = self.DATA_DIR
        
        # 联邦学习配置
        self.num_clients = self.NUM_CLIENTS
        self.alpha_values = self.ALPHA_VALUES
        
        # 第一阶段配置
        self.T_d = self.STAGE1_ROUNDS
        self.stage1_local_epochs = self.STAGE1_LOCAL_EPOCHS
        self.stage1_client_fraction = self.STAGE1_CLIENT_FRACTION
        
        # 第二阶段配置
        self.T_r = self.STAGE2_ROUNDS
        self.stage2_local_epochs = self.STAGE2_LOCAL_EPOCHS
        self.stage2_client_fraction = self.STAGE2_CLIENT_FRACTION
        
        # 训练超参数
        self.batch_size = self.BATCH_SIZE
        self.learning_rate = self.LEARNING_RATE
        self.weight_decay = self.WEIGHT_DECAY
        self.local_epochs = self.STAGE1_LOCAL_EPOCHS  # 默认使用第一阶段的值
        self.use_class_weights = self.USE_CLASS_WEIGHTS  # 类别权重
        
        # 损失函数权重
        self.lambda_vae = self.LAMBDA_VAE
        self.lambda_wgan = self.LAMBDA_WGAN
        self.lambda_proto = self.LAMBDA_PROTO
        self.lambda_ce = self.LAMBDA_CE
        
        # WGAN-GP参数
        self.wgan_critic_iters = self.WGAN_CRITIC_ITERS
        self.wgan_lambda_gp = self.WGAN_LAMBDA_GP
        
        # 差分隐私
        self.epsilon = self.PRIVACY_EPSILON[1]  # 默认使用中等隐私预算
        self.noise_type = self.NOISE_TYPE
        self.delta = self.DELTA
        
        # FedDeProto特定参数
        self.latent_dim = 32  # VAE潜在空间维度
        self.num_rounds_stage1 = self.STAGE1_ROUNDS  # 别名，方便访问
        self.num_rounds_stage2 = self.STAGE2_ROUNDS  # 别名，方便访问
        self.num_classes = 2  # 二分类
        self.aggregation_strategy = 'fedavg'  # 聚合策略
        
        # 阈值配置
        self.accuracy_threshold = self.THRESHOLD_ACC_VARIANCE
        self.similarity_threshold = self.THRESHOLD_COSINE_SIM
        self.stability_rounds = self.THRESHOLD_ACC_WINDOW
        self.acc_fluctuation_threshold = 0.02  # 准确率波动阈值 2%
        self.cosine_sim_threshold = 0.15  # 余弦相似度阈值
        self.stable_rounds = 3  # 需要连续稳定的轮数
        self.adaptive_decay_lambda = self.ADAPTIVE_DECAY_LAMBDA
        
        # FedProx配置
        self.fedprox_mu = 0.1  # 近端项系数（增大到0.1使效果更明显）
        
        # 设备配置
        self.device = self.DEVICE
        self.seed = self.SEED
        
        # 保存与日志
        self.save_dir = self.SAVE_DIR
        self.log_dir = self.LOG_DIR
        self.results_dir = self.RESULTS_DIR
        self.checkpoint_interval = self.CHECKPOINT_INTERVAL
        
        # 可视化
        self.plot_training_curve = self.PLOT_TRAINING_CURVE
        self.plot_confusion_matrix = self.PLOT_CONFUSION_MATRIX
        self.plot_tsne = self.PLOT_TSNE
        self.save_figures = self.SAVE_FIGURES
    
    # ==================== 数据集配置 ====================
    DATASETS = ['australian', 'german', 'xinwang', 'uci']
    DATA_DIR = './data'
    
    # ==================== 联邦学习配置 ====================
    NUM_CLIENTS = 10  # 每个数据集划分的客户端数量
    ALPHA_VALUES = [0.1, 0.3, 1.0]  # LDA的Dirichlet参数（控制异构程度）
    
    # 第一阶段：特征蒸馏
    STAGE1_ROUNDS = 100  # 蒸馏阶段的最大通信轮次（增加到100确保收敛）
    STAGE1_LOCAL_EPOCHS = 5  # 每轮本地训练的epoch数
    STAGE1_CLIENT_FRACTION = 1.0  # 每轮参与训练的客户端比例
    
    # 第二阶段：联邦分类
    STAGE2_ROUNDS = 150  # 分类阶段的最大通信轮次（增加到150确保收敛）
    STAGE2_LOCAL_EPOCHS = 5
    STAGE2_CLIENT_FRACTION = 1.0
    
    # ==================== 训练超参数 ====================
    BATCH_SIZE = 32
    LEARNING_RATE = 0.02  # 提高到0.02以加快收敛
    WEIGHT_DECAY = 1e-4
    USE_CLASS_WEIGHTS = True  # 使用类别权重处理不平衡
    USE_FOCAL_LOSS = True  # 使用Focal Loss处理类别不平衡
    FOCAL_ALPHA = 0.25  # Focal Loss alpha参数
    FOCAL_GAMMA = 2.0   # Focal Loss gamma参数
    
    # 损失函数权重
    LAMBDA_VAE = 1.0      # l1: VAE重建损失
    LAMBDA_WGAN = 1.0     # l2: WGAN-GP对抗损失
    LAMBDA_PROTO = 0.5    # l3: 原型对齐损失
    LAMBDA_CE = 1.0       # l4: 交叉熵损失
    
    # WGAN-GP特定参数
    WGAN_CRITIC_ITERS = 5  # 判别器更新次数
    WGAN_LAMBDA_GP = 10    # 梯度惩罚系数
    
    # ==================== 差分隐私配置 ====================
    PRIVACY_EPSILON = [0.1, 1.0, 10.0]  # 不同隐私预算级别
    NOISE_TYPE = 'laplace'  # 'laplace' 或 'gaussian'
    DELTA = 1e-5  # 高斯机制的delta参数
    
    # ==================== 阈值条件配置 ====================
    THRESHOLD_ACC_WINDOW = 3  # 准确率稳定性检查窗口（连续3轮）
    THRESHOLD_ACC_VARIANCE = 0.02  # 准确率波动阈值（2%）
    THRESHOLD_COSINE_SIM = 0.15  # X_r与X_s余弦相似度上限
    
    # 自适应权重衰减
    ADAPTIVE_DECAY_LAMBDA = 0.5  # 衰减强度系数
    
    # ==================== 设备配置 ====================
    DEVICE = 'cuda'  # 'cuda' 或 'cpu'
    SEED = 42  # 随机种子
    
    # ==================== 保存与日志 ====================
    SAVE_DIR = './checkpoints'
    LOG_DIR = './logs'
    RESULTS_DIR = './results'
    CHECKPOINT_INTERVAL = 10  # 每10轮保存一次checkpoint
    
    # ==================== 可视化配置 ====================
    PLOT_TRAINING_CURVE = True
    PLOT_CONFUSION_MATRIX = True
    PLOT_TSNE = False  # t-SNE降维可视化（耗时）
    SAVE_FIGURES = True
