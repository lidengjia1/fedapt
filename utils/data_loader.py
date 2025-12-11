"""
数据加载模块
支持加载 Australian, German, Xinwang, UCI Credit 四个数据集
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


class CreditDataLoader:
    """信用数据加载器"""
    
    def __init__(self, data_dir='./data', test_size=0.2, random_state=42):
        """
        Args:
            data_dir: 数据文件目录
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.data_dir = data_dir
        self.test_size = test_size
        self.random_state = random_state
        
    def load_dataset(self, dataset_name):
        """
        加载指定数据集
        
        Args:
            dataset_name: 数据集名称 ('australian', 'german', 'xinwang', 'uci')
        
        Returns:
            X_train, X_test, y_train, y_test: 训练和测试数据
        """
        dataset_name = dataset_name.lower()
        scaler = StandardScaler()
        
        if dataset_name == 'australian':
            X_train, X_test, y_train, y_test = self._load_australian(scaler)
        elif dataset_name == 'german':
            X_train, X_test, y_train, y_test = self._load_german(scaler)
        elif dataset_name == 'xinwang':
            X_train, X_test, y_train, y_test = self._load_xinwang(scaler)
        elif dataset_name == 'uci':
            X_train, X_test, y_train, y_test = self._load_uci(scaler)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return X_train, X_test, y_train, y_test
    
    def load_data(self, dataset_name):
        """兼容旧版本的方法"""
        return self.load_dataset(dataset_name)
    
    def _load_australian(self, scaler):
        """加载 Australian Credit 数据集"""
        df = pd.read_csv(f'{self.data_dir}/australian_credit.csv')
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # 标准化
        X = scaler.fit_transform(X)
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def _load_german(self, scaler):
        """加载 German Credit 数据集"""
        df = pd.read_csv(f'{self.data_dir}/german_credit.csv')
        
        # 处理分类特征
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in ['Class', 'default']:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
        
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # 修正标签: German数据集标签为1,2需要转换为0,1
        unique_labels = np.unique(y)
        print(f"German数据集原始标签: {unique_labels}")
        if set(unique_labels) == {1, 2}:
            print("警告: 检测到标签为{1,2}，转换为{0,1}")
            y = y - 1
        elif y.min() > 0:
            print(f"警告: 标签最小值为{y.min()}，调整为从0开始")
            y = y - y.min()
        print(f"转换后标签: {np.unique(y)}，分布: {np.bincount(y)}")
        
        # 标准化
        X = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def _load_xinwang(self, scaler):
        """加载 Xinwang 数据集"""
        df = pd.read_csv(f'{self.data_dir}/xinwang.csv')
        
        # 处理缺失值
        df = df.replace(-99, np.nan)
        df = df.fillna(df.median())
        
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # 标准化
        X = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def _load_uci(self, scaler):
        """加载 UCI Credit 数据集"""
        try:
            df = pd.read_excel(f'{self.data_dir}/uci_credit.xls', header=1)
        except:
            df = pd.read_csv(f'{self.data_dir}/uci_credit.csv')
        
        # 删除ID列
        if 'ID' in df.columns:
            df = df.drop('ID', axis=1)
        
        # 找到标签列（通常是Y或target或最后一列）
        if 'Y' in df.columns:
            label_col = 'Y'
        elif 'target' in df.columns:
            label_col = 'target'
        else:
            label_col = df.columns[-1]
        
        # 提取特征和标签
        X = df.drop(label_col, axis=1).values
        y = df[label_col].values
        
        # 清理异常标签值（字符串等）
        print(f"UCI数据集原始标签类型: {type(y[0])}, 唯一值: {np.unique(y)}")
        y = pd.to_numeric(y, errors='coerce')  # 将非数字转换为NaN
        valid_mask = ~np.isnan(y)
        if not valid_mask.all():
            print(f"警告: 发现{(~valid_mask).sum()}个异常标签，已移除")
            X = X[valid_mask]
            y = y[valid_mask]
        y = y.astype(int)
        
        # 确保标签是0和1
        if y.min() > 0:
            print(f"警告: 标签最小值为{y.min()}，调整为从0开始")
            y = y - y.min()
        print(f"转换后标签: {np.unique(y)}，分布: {np.bincount(y)}")
        
        # 标准化
        X = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
        return X_train, X_test, y_train, y_test
    
    def create_dataloaders(self, X_train, y_train, X_test, y_test, batch_size=64):
        """创建 PyTorch DataLoader"""
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
