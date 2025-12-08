"""
FedDeProto - VAE-WGAN-GP 特征蒸馏模型
包含编码器、解码器、判别器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """VAE编码器 - 将输入映射到潜在空间"""
    
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(Encoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # 输出均值和方差
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """VAE解码器 - 从潜在空间重建输入"""
    
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super(Decoder, self).__init__()
        
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.decoder(z)


class Discriminator(nn.Module):
    """WGAN-GP判别器 - 判断重建样本的真实性"""
    
    def __init__(self, input_dim, hidden_dims):
        super(Discriminator, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.discriminator = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.discriminator(x)


class ResidualClassifier(nn.Module):
    """残差分类器 - 在X_s=X-X_r上进行分类"""
    
    def __init__(self, input_dim, hidden_dims, num_classes):
        super(ResidualClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)


class VAEWGANGP(nn.Module):
    """
    VAE-WGAN-GP 蒸馏模型
    结合VAE的重建能力和WGAN-GP的对抗训练
    """
    
    def __init__(self, config):
        super(VAEWGANGP, self).__init__()
        
        self.input_dim = config['input_dim']
        self.latent_dim = config['latent_dim']
        self.num_classes = config['num_classes']
        
        # 编码器
        self.encoder = Encoder(
            input_dim=self.input_dim,
            hidden_dims=config['encoder_hidden'],
            latent_dim=self.latent_dim
        )
        
        # 解码器（生成器）
        self.decoder = Decoder(
            latent_dim=self.latent_dim,
            hidden_dims=config['decoder_hidden'],
            output_dim=self.input_dim
        )
        
        # 判别器
        self.discriminator = Discriminator(
            input_dim=self.input_dim,
            hidden_dims=config['discriminator_hidden']
        )
        
        # 残差分类器
        self.classifier = ResidualClassifier(
            input_dim=self.input_dim,
            hidden_dims=config['classifier_hidden'],
            num_classes=self.num_classes
        )
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        前向传播
        
        Returns:
            x_recon: 重建特征 X_r
            x_residual: 残差特征 X_s = X - X_r
            mu, logvar: 潜在空间参数
            logits: 分类logits
        """
        # 编码
        mu, logvar = self.encoder(x)
        
        # 重参数化采样
        z = self.reparameterize(mu, logvar)
        
        # 解码（重建）
        x_recon = self.decoder(z)
        
        # 计算残差
        x_residual = x - x_recon
        
        # 在残差上分类
        logits = self.classifier(x_residual)
        
        return x_recon, x_residual, mu, logvar, z, logits
    
    def get_features(self, x):
        """提取可泛化特征（用于第二阶段）"""
        with torch.no_grad():
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decoder(z)
            x_residual = x - x_recon
        return x_residual
    
    def compute_gradient_penalty(self, real_samples, fake_samples, device):
        """计算WGAN-GP的梯度惩罚项"""
        batch_size = real_samples.size(0)
        
        # 随机插值
        alpha = torch.rand(batch_size, 1, device=device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        
        # 判别器输出
        d_interpolates = self.discriminator(interpolates)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates, device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # 梯度惩罚
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty


def test_vae_wgan_gp():
    """测试VAE-WGAN-GP模型"""
    from config.model_configs import ModelConfig
    
    config = ModelConfig.get_config('australian')
    model = VAEWGANGP(config)
    
    # 模拟输入
    batch_size = 32
    x = torch.randn(batch_size, config['input_dim'])
    
    # 前向传播
    x_recon, x_residual, mu, logvar, z, logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {x_recon.shape}")
    print(f"Residual shape: {x_residual.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Logits shape: {logits.shape}")
    
    # 测试判别器
    disc_real = model.discriminator(x)
    disc_fake = model.discriminator(x_recon.detach())
    print(f"Discriminator real: {disc_real.shape}")
    print(f"Discriminator fake: {disc_fake.shape}")
    
    print("\n✓ VAE-WGAN-GP模型测试通过！")


# 为了兼容性，添加别名
VAEWGANDistiller = VAEWGANGP


if __name__ == '__main__':
    test_vae_wgan_gp()
