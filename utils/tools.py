import os
import random
import logging
import torch
import numpy as np

def set_seed(seed=42):
    """固定随机种子保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_logger(log_path):
    """创建日志记录器"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger('HADD')
    logger.setLevel(logging.INFO)

    # 避免重复添加handler
    if logger.handlers:
        logger.handlers.clear()

    # 文件输出
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(console_handler)

    return logger

def save_ckpt(model, optimizer, scheduler, epoch, best_metric, save_path):
    """保存模型检查点"""
    save_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'best_mpjpe': best_metric
    }
    torch.save(save_dict, save_path)

def load_ckpt(ckpt_path, model, optimizer=None, scheduler=None):
    """加载检查点"""
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_mpjpe', 1e9)
    return model, optimizer, scheduler, epoch, best_metric

def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f'Total: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M'

def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
    return path

def tensor2numpy(tensor):
    """Tensor转numpy"""
    return tensor.detach().cpu().numpy()