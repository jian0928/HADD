import torch
import math

def cosine_noise_schedule(timesteps, s=0.008):
    """
    余弦噪声调度 (from "Improved Denoising Diffusion Probabilistic Models")
    Args:
        timesteps: 扩散步数 T
        s: 平滑系数
    Returns:
        alpha: 1 - beta
        alpha_bar: 累乘alpha
        beta: 噪声率
    """
    t = torch.arange(timesteps + 1, dtype=torch.float64)
    f_t = torch.cos((t / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = f_t / f_t[0]

    beta = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    beta = torch.clip(beta, 0.0001, 0.9999)  # 防止数值不稳定

    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    return alpha, alpha_bar, beta

def linear_noise_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    标准线性噪声调度（备用）
    """
    beta = torch.linspace(beta_start, beta_end, timesteps)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return alpha, alpha_bar, beta