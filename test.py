import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入自定义模块
from data.dataset import Human36MDataset, MPI3DHPDataset
from models.hadd import HADD
from utils.metrics import mpjpe, p_mpjpe, pck, auc
from utils.tools import set_seed, create_logger, load_ckpt, ensure_dir, tensor2numpy

# ===================== 测试配置（严格遵循论文） =====================
set_seed(42)  # 固定随机种子保证可复现
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRICS_SAVE_PATH = "./results/metrics/"
POSE_SAVE_PATH = "./results/poses/"
ensure_dir(METRICS_SAVE_PATH)
ensure_dir(POSE_SAVE_PATH)

def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description='HADD Model Testing (Paper Protocol)')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to pre-trained checkpoint')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--high_prec', action='store_true', help='Enable high-precision inference (H=20, Z=10)')
    parser.add_argument('--save_pose', action='store_true', help='Save predicted 3D poses')
    return parser.parse_args()

def load_config(config_path):
    """加载论文配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_epoch(model, test_loader, config, logger, save_pose=False):
    """
    单轮测试（论文标准评估流程）
    返回：所有评估指标的平均值
    """
    model.eval()
    total_mpjpe = []
    total_pmpjpe = []
    total_pck = []
    total_auc = []

    logger.info(f"Start testing | Hypotheses: {config['num_hypotheses']} | Iterations: {config['num_iterations']}")
    pbar = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # 加载数据到设备
            pose_2d = batch['pose_2d'].to(DEVICE, dtype=torch.float32)
            pose_3d_gt = batch['pose_3d'].to(DEVICE, dtype=torch.float32)

            # 模型推理（论文多假设DDIM采样）
            pose_3d_pred, pose_3d_hypo = model(pose_2d=pose_2d, mode='infer')

            # ===================== 计算论文要求的评估指标 =====================
            # 1. MPJPE (mm)
            mpjpe_val = mpjpe(pose_3d_pred, pose_3d_gt).mean().item()
            # 2. P-MPJPE (Procrustes对齐, mm)
            pmpjpe_val = p_mpjpe(pose_3d_pred, pose_3d_gt).mean().item()
            # 3. PCK@100mm (论文阈值)
            pck_val = pck(pose_3d_pred, pose_3d_gt, threshold=config['pck_threshold'])
            # 4. AUC 0-200mm (论文标准)
            auc_val = auc(pose_3d_pred, pose_3d_gt)

            # 累计指标
            total_mpjpe.append(mpjpe_val)
            total_pmpjpe.append(pmpjpe_val)
            total_pck.append(pck_val)
            total_auc.append(auc_val)

            # 进度条显示
            pbar.set_postfix({
                'MPJPE': f'{mpjpe_val:.2f}',
                'P-MPJPE': f'{pmpjpe_val:.2f}',
                'PCK': f'{pck_val:.4f}',
                'AUC': f'{auc_val:.4f}'
            })

            # 保存预测的3D姿态（可选）
            if save_pose:
                np.save(f"{POSE_SAVE_PATH}batch_{batch_idx:04d}_pred.npy", tensor2numpy(pose_3d_pred))
                np.save(f"{POSE_SAVE_PATH}batch_{batch_idx:04d}_gt.npy", tensor2numpy(pose_3d_gt))

    # 计算平均指标
    avg_mpjpe = np.mean(total_mpjpe)
    avg_pmpjpe = np.mean(total_pmpjpe)
    avg_pck = np.mean(total_pck)
    avg_auc = np.mean(total_auc)

    return avg_mpjpe, avg_pmpjpe, avg_pck, avg_auc

def main():
    args = parse_args()
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # 加载配置
    config = load_config(args.config)
    # 高精度模式覆盖配置（论文设定）
    if args.high_prec:
        config['num_hypotheses'] = 20
        config['num_iterations'] = 10
        logger = create_logger(f"{METRICS_SAVE_PATH}test_high_prec.log")
        logger.info("=== HIGH-PRECISION INFERENCE (Paper: H=20, Z=10) ===")
    else:
        logger = create_logger(f"{METRICS_SAVE_PATH}test_light.log")
        logger.info("=== LIGHTWEIGHT INFERENCE (Paper: H=5, Z=1) ===")

    # ===================== 加载测试数据集 =====================
    logger.info(f"Loading dataset: {config['dataset']}")
    if config['dataset'] == 'human36m':
        test_dataset = Human36MDataset(config['data_path'], split='test')
    elif config['dataset'] == 'mpi3dhp':
        test_dataset = MPI3DHPDataset(config['data_path'], split='test')
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")

    # DataLoader（测试：batch_size=8，无shuffle）
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('test_batch_size', 8),
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ===================== 初始化HADD模型 =====================
    logger.info("Initializing HADD model...")
    model = HADD(config).to(DEVICE)
    # 加载预训练权重
    model, _, _, epoch, best_mpjpe = load_ckpt(args.ckpt, model)
    logger.info(f"Loaded checkpoint from epoch {epoch} | Best Val MPJPE: {best_mpjpe:.2f} mm")

    # ===================== 执行测试 =====================
    logger.info("=" * 60)
    logger.info("Testing Start (Paper Protocol)")
    logger.info("=" * 60)
    avg_mpjpe, avg_pmpjpe, avg_pck, avg_auc = test_epoch(
        model=model,
        test_loader=test_loader,
        config=config,
        logger=logger,
        save_pose=args.save_pose
    )

    # ===================== 输出最终结果（论文格式） =====================
    logger.info("=" * 60)
    logger.info("FINAL TEST RESULTS (Paper Metrics)")
    logger.info("=" * 60)
    logger.info(f"Dataset: {config['dataset']}")
    logger.info(f"Sequence Length: {config['seq_len']} | Joints: {config['num_joints']}")
    logger.info(f"MPJPE (mm): {avg_mpjpe:.2f}")
    logger.info(f"P-MPJPE (mm): {avg_pmpjpe:.2f}")
    logger.info(f"PCK@{config['pck_threshold']}mm: {avg_pck:.4f}")
    logger.info(f"AUC (0-200mm): {avg_auc:.4f}")
    logger.info("=" * 60)

    # 保存结果到文件
    result_file = f"{METRICS_SAVE_PATH}{config['dataset']}_results.txt"
    with open(result_file, 'w') as f:
        f.write(f"=== HADD Test Results ===\n")
        f.write(f"Dataset: {config['dataset']}\n")
        f.write(f"Inference Mode: {'High-Precision' if args.high_prec else 'Lightweight'}\n")
        f.write(f"MPJPE: {avg_mpjpe:.2f} mm\n")
        f.write(f"P-MPJPE: {avg_pmpjpe:.2f} mm\n")
        f.write(f"PCK@{config['pck_threshold']}mm: {avg_pck:.4f}\n")
        f.write(f"AUC: {avg_auc:.4f}\n")

    logger.info(f"Results saved to: {result_file}")
    logger.info("Testing finished successfully!")

if __name__ == '__main__':
    main()