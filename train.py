import os
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.dataset import Human36MDataset, MPI3DHPDataset
from models.hadd import HADD
from utils.tools import set_seed, create_logger, save_ckpt

# Set random seed for reproducibility
set_seed(42)

def main(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Create logger and ckpt dir
    logger = create_logger('./logs/train.log')
    os.makedirs('./ckpts', exist_ok=True)
    # Device
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info(f"Loading dataset: {config['dataset']}")
    if config['dataset'] == 'human36m':
        train_dataset = Human36MDataset(config['data_path'], split='train', seq_len=config['seq_len'])
        val_dataset = Human36MDataset(config['data_path'], split='val', seq_len=config['seq_len'])
    elif config['dataset'] == 'mpi3dhp':
        train_dataset = MPI3DHPDataset(config['data_path'], split='train', seq_len=config['seq_len'])
        val_dataset = MPI3DHPDataset(config['data_path'], split='val', seq_len=config['seq_len'])
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    logger.info("Initializing HADD model")
    model = HADD(config).to(device)
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], betas=(0.9, 0.999), eps=1e-8)
    # LR Scheduler (cosine annealing with warm-up)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=config['warmup_epochs'] * len(train_loader))
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'] * len(train_loader) - config['warmup_epochs'] * len(train_loader))
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[config['warmup_epochs'] * len(train_loader)])
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=config['mixed_precision'])

    # Training loop
    logger.info(f"Start training for {config['epochs']} epochs")
    best_val_mpjpe = float('inf')
    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} (Train)")
        for batch in pbar:
            pose_2d = batch['pose_2d'].to(device, dtype=torch.float32)
            pose_3d_gt = batch['pose_3d'].to(device, dtype=torch.float32)
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=config['mixed_precision']):
                loss, loss_dict = model(pose_3d_gt=pose_3d_gt, pose_2d=pose_2d, mode='train')
            # Backward pass (gradient accumulation)
            scaler.scale(loss / config['gradient_accumulation']).backward()
            # Gradient clipping
            if (pbar.n + 1) % config['gradient_accumulation'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            # Log
            train_loss += loss.item()
            pbar.set_postfix({'Loss': loss_dict['total_loss'], 'Ldis': loss_dict['ldis'], 'Lpos': loss_dict['lpos']})
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} | Avg Train Loss: {avg_train_loss:.6f}")

        # Validation
        model.eval()
        val_mpjpe = 0.0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} (Val)")
        with torch.no_grad():
            for batch in pbar_val:
                pose_2d = batch['pose_2d'].to(device, dtype=torch.float32)
                pose_3d_gt = batch['pose_3d'].to(device, dtype=torch.float32)
                # Inference
                pose_3d_pred, _ = model(pose_2d=pose_2d, mode='infer')
                # Calculate MPJPE
                from utils.metrics import mpjpe
                batch_mpjpe = mpjpe(pose_3d_pred, pose_3d_gt).mean().item()
                val_mpjpe += batch_mpjpe
                pbar_val.set_postfix({'Val MPJPE': batch_mpjpe})
        avg_val_mpjpe = val_mpjpe / len(val_loader)
        logger.info(f"Epoch {epoch+1} | Avg Val MPJPE: {avg_val_mpjpe:.2f} mm")

        # Save best model
        if avg_val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = avg_val_mpjpe
            save_ckpt(model, optimizer, scheduler, epoch, best_val_mpjpe, f"./ckpts/hadd_{config['dataset']}_best.pth")
            logger.info(f"Save best model | Val MPJPE: {best_val_mpjpe:.2f} mm")

    logger.info(f"Training finished | Best Val MPJPE: {best_val_mpjpe:.2f} mm")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')