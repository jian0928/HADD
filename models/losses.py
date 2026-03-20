import torch
import torch.nn as nn
import torch.nn.functional as F

class DisentanglementLoss(nn.Module):
    def __init__(self, num_bones=16):
        super().__init__()
        self.num_bones = num_bones

    def forward(self, pred_len, pred_dir, gt_len, gt_dir):
        """
        3D Disentanglement Loss (Ldis = Ll + Ld)
        Args:
            pred_len/pred_dir: (B, N, K, 1/3) - Predicted bone length/direction
            gt_len/gt_dir: (B, N, K, 1/3) - Ground-truth bone length/direction
        Returns:
            ldis: Scalar - Total disentanglement loss
            ll: Scalar - Bone length loss (L2)
            ld: Scalar - Bone direction loss (L2)
        """
        B, N, K, _ = pred_len.shape
        # Bone Length Loss (Ll) - L2 norm, normalized by B*N*K
        ll = F.mse_loss(pred_len, gt_len, reduction='sum') / (B * N * K)
        # Bone Direction Loss (Ld) - L2 norm for unit vector, normalized by B*N*K
        ld = F.mse_loss(pred_dir, gt_dir, reduction='sum') / (B * N * K)
        # Total disentanglement loss
        ldis = ll + ld
        return ldis, ll, ld

class PoseLoss(nn.Module):
    def __init__(self, num_joints=17):
        super().__init__()
        self.num_joints = num_joints

    def forward(self, pred_pose, gt_pose):
        """
        3D Pose Loss (Lpos) - L2 norm for 3D joint coordinates
        Args:
            pred_pose/gt_pose: (B, N, J, 3) - Predicted/GT 3D pose
        Returns:
            lpos: Scalar - Pose loss (normalized by B*N*J*3)
        """
        B, N, J, _ = pred_pose.shape
        lpos = F.mse_loss(pred_pose, gt_pose, reduction='sum') / (B * N * J * 3)
        return lpos

class HybridLoss(nn.Module):
    def __init__(self, num_joints=17, num_bones=16):
        super().__init__()
        self.ldis_loss = DisentanglementLoss(num_bones)
        self.lpos_loss = PoseLoss(num_joints)

    def forward(self, pred_len, pred_dir, gt_len, gt_dir, pred_pose, gt_pose):
        """
        Hybrid Loss (L = Ldis + Lpos) - equal weight as in the paper
        Args:
            pred_len/pred_dir: Predicted bone features
            gt_len/gt_dir: GT bone features
            pred_pose/gt_pose: Predicted/GT 3D pose
        Returns:
            total_loss: Scalar - Hybrid loss
            loss_dict: Dict - Detailed loss values
        """
        ldis, ll, ld = self.ldis_loss(pred_len, pred_dir, gt_len, gt_dir)
        lpos = self.lpos_loss(pred_pose, gt_pose)
        # Total loss (equal weight)
        total_loss = ldis + lpos
        # Loss dict for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'ldis': ldis.item(),
            'll': ll.item(),
            'ld': ld.item(),
            'lpos': lpos.item()
        }
        return total_loss, loss_dict