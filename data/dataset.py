import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm

# ===================== 论文固定超参数 =====================
SEQ_LEN = 243          # 论文：固定输入时序长度 N=243
NUM_JOINTS = 17        # 论文：17个标准关节
ROOT_JOINT = 0         # 论文：骨盆 (Pelvis) 作为根关节
NORMALIZE_RANGE = 1.0  # 论文：坐标归一化范围

# ===================== Human3.6M 预处理（论文实验） =====================
def process_human36m(raw_h5_path, save_h5_path):
    print("[HADD] 开始预处理 Human3.6M 数据集 (严格遵循论文实验)")
    with h5py.File(raw_h5_path, "r") as f:
        pose_2d = f["2d_keypoints"][:]  # 论文：OpenPose 提取的2D关键点
        pose_3d = f["3d_pose"][:]       # 论文：官方3D真实姿态
        subjects = f["subject"][:]      # 受试者ID

    # 论文：仅保留17个核心关节
    pose_2d = pose_2d[..., :NUM_JOINTS, :]
    pose_3d = pose_3d[..., :NUM_JOINTS, :]

    processed_2d = []
    processed_3d = []
    train_ids, val_ids, test_ids = [], [], []

    for idx in tqdm(range(len(pose_2d)), desc="Processing samples"):
        # ===================== 论文核心预处理1：根中心化 =====================
        root_3d = pose_3d[idx:idx+1, :, ROOT_JOINT:ROOT_JOINT+1, :]  # 骨盆坐标
        pose_3d_centered = pose_3d[idx] - root_3d  # 去除全局平移

        root_2d = pose_2d[idx:idx+1, :, ROOT_JOINT:ROOT_JOINT+1, :]
        pose_2d_centered = pose_2d[idx] - root_2d

        # ===================== 论文核心预处理2：固定序列长度 =====================
        pose_2d_fixed = _resize_sequence(pose_2d_centered, SEQ_LEN)
        pose_3d_fixed = _resize_sequence(pose_3d_centered, SEQ_LEN)

        # ===================== 论文核心预处理3：坐标归一化 =====================
        pose_2d_norm = pose_2d_fixed / np.max(np.abs(pose_2d_fixed)) * NORMALIZE_RANGE
        pose_3d_norm = pose_3d_fixed / 1000.0  # 毫米 → 米（论文标准单位）

        processed_2d.append(pose_2d_norm)
        processed_3d.append(pose_3d_norm)

        # ===================== 论文数据集划分 =====================
        subj = subjects[idx].decode()
        if subj in ["S1", "S5", "S6", "S7", "S8"]:
            train_ids.append(idx)
        elif subj == "S9":
            val_ids.append(idx)
        elif subj == "S11":
            test_ids.append(idx)

    # 保存预处理后的数据
    with h5py.File(save_h5_path, "w") as f:
        f.create_dataset("pose_2d", data=np.array(processed_2d, dtype=np.float32))
        f.create_dataset("pose_3d", data=np.array(processed_3d, dtype=np.float32))
        f.create_dataset("train", data=np.array(train_ids, dtype=np.int32))
        f.create_dataset("val", data=np.array(val_ids, dtype=np.int32))
        f.create_dataset("test", data=np.array(test_ids, dtype=np.int32))

    print(f"[HADD] Human3.6M 预处理完成 | 训练集:{len(train_ids)} 验证集:{len(val_ids)} 测试集:{len(test_ids)}")

# ===================== MPI-INF-3DHP 预处理（论文实验） =====================
def process_mpi3dhp(raw_h5_path, save_h5_path):
    print("[HADD] 开始预处理 MPI-INF-3DHP 数据集")
    with h5py.File(raw_h5_path, "r") as f:
        pose_2d = f["2d_keypoints"][:]
        pose_3d = f["3d_pose"][:]

    processed_2d, processed_3d = [], []
    for idx in tqdm(range(len(pose_2d))):
        # 根中心化
        pose_3d[idx] -= pose_3d[idx, :, ROOT_JOINT:ROOT_JOINT+1, :]
        pose_2d[idx] -= pose_2d[idx, :, ROOT_JOINT:ROOT_JOINT+1, :]
        # 固定序列长度
        processed_2d.append(_resize_sequence(pose_2d[idx], SEQ_LEN))
        processed_3d.append(_resize_sequence(pose_3d[idx], SEQ_LEN))

    # 保存
    with h5py.File(save_h5_path, "w") as f:
        f.create_dataset("pose_2d", data=np.array(processed_2d, dtype=np.float32))
        f.create_dataset("pose_3d", data=np.array(processed_3d, dtype=np.float32))

    print("[HADD] MPI-INF-3DHP 预处理完成")

# ===================== 工具函数：序列裁剪/填充 =====================
def _resize_sequence(seq, target_len):
    cur_len = seq.shape[0]
    if cur_len > target_len:
        return seq[:target_len]
    pad = np.zeros((target_len - cur_len, NUM_JOINTS, seq.shape[-1]), dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HADD 论文数据集预处理脚本")
    parser.add_argument("--dataset", type=str, required=True, choices=["human36m", "mpi3dhp"])
    parser.add_argument("--raw_path", type=str, required=True, help="原始H5数据路径")
    parser.add_argument("--save_path", type=str, required=True, help="预处理后保存路径")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    if args.dataset == "human36m":
        process_human36m(args.raw_path, args.save_path)
    else:
        process_mpi3dhp(args.raw_path, args.save_path)