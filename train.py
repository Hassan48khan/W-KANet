"""
Training script for W-KANet
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os

from W_KANet import WKANet
from loss import TaskAdaptiveLoss


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        
        # Compute loss
        loss_dict = criterion(outputs, masks)
        loss = loss_dict['total']
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Update progress
        total_loss += loss.item()
        pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validation"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss_dict = criterion(outputs, masks)
            total_loss += loss_dict['total'].item()
    
    return total_loss / len(dataloader)


def main(args):
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = WKANet(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        base_channels=32
    ).to(device)
    
    # Loss
    criterion = TaskAdaptiveLoss(model, num_classes=args.num_classes)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # TODO: Add your dataloader here
    # train_loader = DataLoader(...)
    # val_loader = DataLoader(...)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Train Loss: {train_loss:.4f}')
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f'Saved best model (val_loss: {val_loss:.4f})')
        
        # Scheduler step
        scheduler.step()
    
    print(f'\nTraining complete! Best val loss: {best_val_loss:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Model
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # Data
    parser.add_argument('--dataset', type=str, default='camus')
    parser.add_argument('--data_path', type=str, required=True)
    
    # Misc
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
