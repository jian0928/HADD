import torch
import torch.nn as nn
from utils.noise_schedule import cosine_noise_schedule

class DiffusionProcess(nn.Module):
    def __init__(self, diffusion_steps=1000, noise_schedule='cosine'):
        super().__init__()
        self.T = diffusion_steps
        # Get noise schedule (alpha, alpha_bar, beta)
        self.alpha, self.alpha_bar, self.beta = cosine_noise_schedule(self.T)
        self.alpha = self.alpha.to(torch.float32)
        self.alpha_bar = self.alpha_bar.to(torch.float32)
        self.beta = self.beta.to(torch.float32)

    def forward_diffusion(self, x0, t):
        """
        Forward diffusion: x0 -> xt with noise at step t
        Args:
            x0: (B, N, K, D) - Clean data (bone_len: D=1; bone_dir: D=3)
            t: (B,) - Diffusion step
        Returns:
            xt: (B, N, K, D) - Noisy data
            eps: (B, N, K, D) - Gaussian noise
        """
        B = x0.shape[0]
        alpha_bar_t = self.alpha_bar[t].view(B, 1, 1, 1)
        eps = torch.randn_like(x0)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps
        return xt, eps

    def ddim_sampling(self, hstd_model, xt, t, t_prev, cond, num_hypotheses=5):
        """
        DDIM sampling for backward diffusion (multi-hypothesis)
        Args:
            hstd_model: HSTD module for denoising
            xt: (B*H, N, K, 4) - Concatenated bone_len + bone_dir (noisy)
            t, t_prev: Diffusion steps
            cond: (B*H, N, J, 2) - 2D pose condition
            num_hypotheses: H - number of hypotheses
        Returns:
            x_prev: (B*H, N, K, 4) - Denoised data at step t_prev
        """
        B = xt.shape[0] // num_hypotheses
        alpha_t = self.alpha[t].view(B*H, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(B*H, 1, 1, 1)
        alpha_bar_t_prev = self.alpha_bar[t_prev].view(B*H, 1, 1, 1)

        # Denoise with HSTD model (condition on 2D pose)
        eps_theta = hstd_model(xt, cond)
        # DDIM update rule
        x0_hat = (xt - torch.sqrt(1 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_bar_t)
        x0_hat = torch.clamp(x0_hat, -1.0, 1.0)  # Stability clamp
        x_prev = torch.sqrt(alpha_bar_t_prev) * x0_hat + torch.sqrt(1 - alpha_bar_t_prev) * eps_theta
        return x_prev