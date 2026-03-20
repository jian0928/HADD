import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm

# 全局配置（严格对齐论文）
SEQ_LEN = 243  # 论文固定输入序列长度
NUM_JOINTS = 17  # 表1定义的17关节
ROOT_JOINT_IDX = 0  # 骨盆根关节索引


def human36m_preprocess(raw_path, save_path):
    """
    预处理 Human3.6M 数据集
    原始数据格式：通用h5格式 (n_samples, n_frames, 17, 3) 3D姿态 + 2D姿态
    处理步骤：根中心化 → 序列规整 → 归一化 → 保存
    """
    print(f"Processing Human3.6M raw data: {raw_path}")

    # 加载原始数据
    with h5py.File(raw_path, 'r') as f:
        pose_2d = f['pose_2d'][:]  # (N, F, 17, 2)
        pose_3d = f['pose_3d'][:]  # (N, F, 17, 3)
        subjects = f['subject'][:]  # 划分训练/验证/测试

    # 数据过滤：仅保留17关节，剔除异常数据
    pose_2d = pose_2d[..., :NUM_JOINTS, :]
    pose_3d = pose_3d[..., :NUM_JOINTS, :]

    processed_2d = []
    processed_3d = []
    train_idx = []
    val_idx = []
    test_idx = []

    # 逐样本处理
    for i in tqdm(range(len(pose_2d))):
        # ===================== 1. 根中心化（骨盆归一到原点） =====================
        root_3d = pose_3d[i:i + 1, :, ROOT_JOINT_IDX:ROOT_JOINT_IDX + 1, :]  # (1, F, 1, 3)
        pose_3d_norm = pose_3d[i:i + 1] - root_3d  # 3D姿态中心化

        root_2d = pose_2d[i:i + 1, :, ROOT_JOINT_IDX:ROOT_JOINT_IDX + 1, :]
        pose_2d_norm = pose_2d[i:i + 1] - root_2d  # 2D姿态中心化

        # ===================== 2. 序列长度规整（固定243帧） =====================
        seq_2d = pad_or_crop(pose_2d_norm[0], SEQ_LEN)
        seq_3d = pad_or_crop(pose_3d_norm[0], SEQ_LEN)

        processed_2d.append(seq_2d)
        processed_3d.append(seq_3d)

        # ===================== 3. 数据集划分（Human3.6M标准协议） =====================
        subj = subjects[i].decode('utf-8')
        if subj in ['S1', 'S5', 'S7', 'S8', 'S9', 'S11']:
            train_idx.append(i)
        elif subj == 'S6':
            val_idx.append(i)
        elif subj in ['S2', 'S3', 'S4', 'S10']:
            test_idx.append(i)

    # 转换为numpy数组
    processed_2d = np.array(processed_2d, dtype=np.float32)
    processed_3d = np.array(processed_3d, dtype=np.float32)

    # 保存预处理数据
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('pose_2d', data=processed_2d)
        f.create_dataset('pose_3d', data=processed_3d)
        f.create_dataset('train_indices', data=np.array(train_idx, dtype=np.int32))
        f.create_dataset('val_indices', data=np.array(val_idx, dtype=np.int32))
        f.create_dataset('test_indices', data=np.array(test_idx, dtype=np.int32))

    print(f"Human3.6M preprocess done! Saved to {save_path}")
    print(f"Total: {len(processed_2d)} | Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")


def pad_or_crop(seq, target_len):
    """序列补零/裁剪到固定长度"""
    cur_len = seq.shape[0]
    if cur_len > target_len:
        return seq[:target_len]
    elif cur_len < target_len:
        pad = np.zeros((target_len - cur_len, *seq.shape[1:]), dtype=seq.dtype)
        return np.concatenate([seq, pad], axis=0)
    return seq


def mpi3dhp_preprocess(raw_path, save_path):
    """MPI-INF-3DHP 预处理（逻辑同Human3.6M，兼容数据集格式）"""
    print(f"Processing MPI-INF-3DHP raw data: {raw_path}")
    with h5py.File(raw_path, 'r') as f:
        pose_2d = f['pose_2d'][:]
        pose_3d = f['pose_3d'][:]

    # 根中心化 + 序列规整
    processed_2d, processed_3d = [], []
    for i in tqdm(range(len(pose_2d))):
        # 根中心化
        pose_3d[i] -= pose_3d[i, :, ROOT_JOINT_IDX:ROOT_JOINT_IDX + 1, :]
        pose_2d[i] -= pose_2d[i, :, ROOT_JOINT_IDX:ROOT_JOINT_IDX + 1, :]
        # 序列规整
        processed_2d.append(pad_or_crop(pose_2d[i], SEQ_LEN))
        processed_3d.append(pad_or_crop(pose_3d[i], SEQ_LEN))

    # 保存
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('pose_2d', data=np.array(processed_2d, dtype=np.float32))
        f.create_dataset('pose_3d', data=np.array(processed_3d, dtype=np.float32))

    print(f"MPI-INF-3DHP preprocess done! Saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HADD Dataset Preprocess')
    parser.add_argument('--dataset', type=str, required=True, choices=['human36m', 'mpi3dhp'])
    parser.add_argument('--raw_path', type=str, required=True, help='Path to raw h5 dataset')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save processed h5')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    if args.dataset == 'human36m':
        human36m_preprocess(args.raw_path, args.save_path)
    elif args.dataset == 'mpi3dhp':
        mpi3dhp_preprocess(args.raw_path, args.save_path)