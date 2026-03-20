import torch
import numpy as np


def mpjpe(pred, gt):
    """
    计算平均每关节位置误差 (Mean Per Joint Position Error)
    Args:
        pred: (B, N, J, 3) 预测3D坐标
        gt: (B, N, J, 3) 真实3D坐标
    Return:
        mpjpe: (B, N) 每个样本每帧的平均误差 (mm)
    """
    error = torch.sqrt(torch.sum((pred - gt) ** 2, dim=-1))  # (B, N, J)
    return error.mean(dim=-1)  # (B, N)


def p_mpjpe(pred, gt):
    """
    计算校准后的MPJPE (Procrustes Aligned MPJPE) - 消除全局旋转/尺度偏移
    Args:
        pred: (B, N, J, 3)
        gt: (B, N, J, 3)
    Return:
        p_mpjpe: (B, N)
    """
    B, N, J, _ = pred.shape
    pred_aligned = pred.clone()

    for b in range(B):
        for n in range(N):
            # 提取单帧姿态
            p = pred[b, n].cpu().numpy()  # (J,3)
            g = gt[b, n].cpu().numpy()  # (J,3)

            # 中心化
            p_cent = p - p.mean(axis=0)
            g_cent = g - g.mean(axis=0)

            # SVD求解旋转矩阵
            H = p_cent.T @ g_cent
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T

            # 特殊情况处理（反射解）
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            # 应用旋转
            aligned = p_cent @ R
            pred_aligned[b, n] = torch.from_numpy(aligned + g.mean(axis=0)).to(pred.device)

    return mpjpe(pred_aligned, gt)


def pck(pred, gt, threshold=100):
    """
    计算正确关键点百分比 (Percentage of Correct Keypoints)
    Args:
        threshold: 误差阈值 (mm)
    Return:
        pck_score: 0~1
    """
    error = torch.sqrt(torch.sum((pred - gt) ** 2, dim=-1))  # (B, N, J)
    correct = (error < threshold).float()
    return correct.mean().item()


def auc(pred, gt, thresholds=None):
    """
    计算AUC值 (Area Under Curve)
    Args:
        thresholds: 阈值列表，默认 [0,20,...,200]
    """
    if thresholds is None:
        thresholds = np.arange(0, 201, 20)

    error = torch.sqrt(torch.sum((pred - gt) ** 2, dim=-1)).cpu().numpy()
    auc_scores = []

    for th in thresholds:
        correct = (error < th).mean()
        auc_scores.append(correct)

    return np.trapz(auc_scores, thresholds) / thresholds[-1]


def mpjpe_2d_proj(pred_3d, pose_2d, intrinsic=None):
    """
    3D姿态投影到2D后计算误差（用于推理选择最优假设）
    Args:
        pred_3d: (B, H, N, J, 3)
        pose_2d: (B, H, N, J, 2)
    Return:
        proj_error: (B, H)
    """
    if intrinsic is None:
        # 默认简化投影
        proj_2d = pred_3d[..., :2] / (pred_3d[..., 2:3] + 1e-6)
    else:
        fx, fy, cx, cy = intrinsic
        proj_2d = torch.zeros_like(pose_2d)
        proj_2d[..., 0] = fx * pred_3d[..., 0] / (pred_3d[..., 2] + 1e-6) + cx
        proj_2d[..., 1] = fy * pred_3d[..., 1] / (pred_3d[..., 2] + 1e-6) + cy

    error = torch.sqrt(torch.sum((proj_2d - pose_2d) ** 2, dim=-1))  # (B, H, N, J)
    return error.mean(dim=-1).mean(dim=-1)  # (B, H)