import torch
import torch.nn as nn
from models.disentangle import PoseDisentangler
from models.diffusion import DiffusionProcess
from models.hstd import HSTD
from models.losses import HybridLoss

class HADD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 3D Pose Disentangler
        self.disentangler = PoseDisentangler(
            num_joints=config['num_joints'],
            num_bones=config['num_bones']
        )
        # Diffusion Process
        self.diffusion = DiffusionProcess(
            diffusion_steps=config['diffusion_steps'],
            noise_schedule=config['noise_schedule']
        )
        # HSTD Module
        self.hstd = HSTD(
            num_joints=config['num_joints'],
            num_frames=config['seq_len'],
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads'],
            ffn_dim=config['ffn_dim'],
            dropout=config['dropout'],
            num_hstd_layers=config['num_hstd_layers'],
            hier_emb_dim=config['hier_emb_dim']
        )
        # Hybrid Loss Function
        self.loss_fn = HybridLoss(
            num_joints=config['num_joints'],
            num_bones=config['num_bones']
        )

    def forward_train(self, pose_3d_gt, pose_2d):
        """
        Training forward pass
        Args:
            pose_3d_gt: (B, N, J, 3) - GT 3D pose
            pose_2d: (B, N, J, 2) - 2D pose condition
        Returns:
            total_loss: Scalar - Hybrid loss
            loss_dict: Dict - Detailed loss values
        """
        # Step 1: Disentangle GT 3D pose to bone length/direction
        bone_len_gt, bone_dir_gt = self.disentangler(pose_3d_gt)
        # Step 2: Concatenate bone features (for diffusion)
        bone_feat_gt = torch.cat([bone_len_gt, bone_dir_gt], dim=-1)  # (B, N, K, 4)
        # Step 3: Random diffusion step t
        B = pose_3d_gt.shape[0]
        t = torch.randint(0, self.config['diffusion_steps'], (B,), device=pose_3d_gt.device)
        # Step 4: Forward diffusion - add noise to bone features
        bone_feat_t, eps_gt = self.diffusion.forward_diffusion(bone_feat_gt, t)
        # Step 5: HSTD denoising - predict noise
        eps_pred = self.hstd(bone_feat_t, pose_2d)
        # Step 6: Reconstruct bone features and 3D pose from predicted noise
        alpha_bar_t = self.diffusion.alpha_bar[t].view(B, 1, 1, 1)
        bone_feat_pred = (bone_feat_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        bone_len_pred = bone_feat_pred[..., :1]
        bone_dir_pred = bone_feat_pred[..., 1:]
        pose_3d_pred = self.disentangler.reconstruct(bone_len_pred, bone_dir_pred)
        # Step 7: Calculate hybrid loss
        total_loss, loss_dict = self.loss_fn(
            pred_len=bone_len_pred, pred_dir=bone_dir_pred,
            gt_len=bone_len_gt, gt_dir=bone_dir_gt,
            pred_pose=pose_3d_pred, gt_pose=pose_3d_gt
        )
        return total_loss, loss_dict

    @torch.no_grad()
    def forward_infer(self, pose_2d):
        """
        Inference forward pass (multi-hypothesis DDIM sampling)
        Args:
            pose_2d: (B, N, J, 2) - 2D pose condition
        Returns:
            pose_3d_pred: (B, N, J, 3) - Final 3D pose prediction
            pose_3d_hypo: (B, H, N, J, 3) - All hypothesis predictions
        """
        B, N, J, _ = pose_2d.shape
        H = self.config['num_hypotheses']
        T = self.config['diffusion_steps']
        Z = self.config['num_iterations']
        # Expand 2D condition for multi-hypothesis: (B, N, J, 2) -> (B*H, N, J, 2)
        pose_2d_hypo = pose_2d.unsqueeze(1).expand(B, H, N, J, 2).reshape(B*H, N, J, 2)
        # Initial noisy bone feature (pure Gaussian noise)
        bone_feat_xt = torch.randn(B*H, N, self.config['num_bones'], 4, device=pose_2d.device)
        # Multi-iteration DDIM sampling
        for z in range(Z):
            # DDIM reverse diffusion from T-1 to 0
            for t in range(T-1, -1, -1):
                t_prev = max(t - 1, 0)
                t_tensor = torch.tensor([t], device=pose_2d.device).expand(B*H)
                t_prev_tensor = torch.tensor([t_prev], device=pose_2d.device).expand(B*H)
                # DDIM sampling step
                bone_feat_xt = self.diffusion.ddim_sampling(
                    hstd_model=self.hstd,
                    xt=bone_feat_xt,
                    t=t_tensor,
                    t_prev=t_prev_tensor,
                    cond=pose_2d_hypo,
                    num_hypotheses=H
                )
        # Get all hypothesis bone features and reconstruct 3D pose
        bone_feat_pred = bone_feat_xt.reshape(B, H, N, self.config['num_bones'], 4)
        bone_len_pred = bone_feat_pred[..., :1]
        bone_dir_pred = bone_feat_pred[..., 1:]
        pose_3d_hypo = torch.zeros(B, H, N, J, 3, device=pose_2d.device)
        for h in range(H):
            pose_3d_hypo[:, h, :, :, :] = self.disentangler.reconstruct(bone_len_pred[:, h, :, :, :], bone_dir_pred[:, h, :, :, :])
        # Select best hypothesis (minimum MPJPE to 2D pose)
        from utils.metrics import mpjpe_2d_proj
        mpjpe_vals = mpjpe_2d_proj(pose_3d_hypo, pose_2d.unsqueeze(1).expand(B, H, N, J, 2))  # (B, H)
        best_hypo_idx = torch.argmin(mpjpe_vals, dim=1)  # (B,)
        # Gather best hypothesis
        pose_3d_pred = torch.zeros(B, N, J, 3, device=pose_2d.device)
        for b in range(B):
            pose_3d_pred[b, :, :, :] = pose_3d_hypo[b, best_hypo_idx[b], :, :, :]
        return pose_3d_pred, pose_3d_hypo

    def forward(self, pose_3d_gt=None, pose_2d=None, mode='train'):
        """
        Unified forward pass
        Args:
            pose_3d_gt: GT 3D pose (train only)
            pose_2d: 2D pose condition
            mode: 'train' / 'infer'
        Returns:
            Train: total_loss, loss_dict
            Infer: pose_3d_pred, pose_3d_hypo
        """
        if mode == 'train':
            assert pose_3d_gt is not None, "GT 3D pose required for training"
            return self.forward_train(pose_3d_gt, pose_2d)
        elif mode == 'infer':
            return self.forward_infer(pose_2d)
        else:
            raise ValueError(f"Invalid mode: {mode} (only 'train'/'infer' supported)")