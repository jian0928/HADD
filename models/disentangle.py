import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.skeleton import SKELETON_HUMAN36M  # 17 joints + parent-child mapping

class PoseDisentangler(nn.Module):
    def __init__(self, num_joints=17, num_bones=16):
        super().__init__()
        self.num_joints = num_joints
        self.num_bones = num_bones
        self.parent = SKELETON_HUMAN36M['parent']  # Parent joint index for each bone

    def forward(self, pose_3d):
        """
        Disentangle 3D pose to bone length and bone direction
        Args:
            pose_3d: (B, N, J, 3) - Batch, Frames, Joints, 3D Coords
        Returns:
            bone_len: (B, N, K, 1) - Bone length (K=J-1)
            bone_dir: (B, N, K, 3) - Bone direction (unit vector)
        """
        B, N, J, _ = pose_3d.shape
        bone_len = torch.zeros(B, N, self.num_bones, 1, device=pose_3d.device)
        bone_dir = torch.zeros(B, N, self.num_bones, 3, device=pose_3d.device)

        for k in range(self.num_bones):
            c_joint = k + 1  # Child joint
            p_joint = self.parent[k]  # Parent joint
            bone_vec = pose_3d[:, :, c_joint, :] - pose_3d[:, :, p_joint, :]
            # Bone length (L2 norm)
            bl = torch.norm(bone_vec, dim=-1, keepdim=True)
            # Bone direction (unit vector)
            bd = F.normalize(bone_vec, p=2, dim=-1)
            bone_len[:, :, k, :] = bl
            bone_dir[:, :, k, :] = bd

        return bone_len, bone_dir

    def reconstruct(self, bone_len, bone_dir):
        """
        Reconstruct 3D pose from bone length and direction (forward kinematics)
        Args:
            bone_len: (B, N, K, 1)
            bone_dir: (B, N, K, 3)
        Returns:
            pose_3d: (B, N, J, 3)
        """
        B, N, K, _ = bone_len.shape
        pose_3d = torch.zeros(B, N, self.num_joints, 3, device=bone_len.device)
        # Pelvis (root joint) is fixed at origin
        for k in range(self.num_bones):
            c_joint = k + 1
            p_joint = self.parent[k]
            pose_3d[:, :, c_joint, :] = pose_3d[:, :, p_joint, :] + bone_len[:, :, k, :] * bone_dir[:, :, k, :]
        return pose_3d