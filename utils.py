"""
Utility functions for W-KANet
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_dice(pred, target, threshold=0.5):
    """
    Compute Dice coefficient
    
    Args:
        pred: Predictions (B, 1, H, W) or (B, H, W)
        target: Ground truth (B, 1, H, W) or (B, H, W)
        threshold: Binarization threshold
    Returns:
        Dice score (float)
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    
    return dice.item()


def compute_iou(pred, target, threshold=0.5):
    """
    Compute IoU (Intersection over Union)
    
    Args:
        pred: Predictions
        target: Ground truth
        threshold: Binarization threshold
    Returns:
        IoU score (float)
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + 1e-7) / (union + 1e-7)
    
    return iou.item()


def compute_hd95(pred, target, threshold=0.5, spacing=[1.0, 1.0]):
    """
    Compute 95th percentile Hausdorff Distance
    
    Args:
        pred: Predictions
        target: Ground truth
        threshold: Binarization threshold
        spacing: Pixel spacing [height, width]
    Returns:
        HD95 in mm (float)
    """
    from scipy.spatial.distance import directed_hausdorff
    
    pred = (torch.sigmoid(pred) > threshold).cpu().numpy()
    target = target.cpu().numpy()
    
    # Get boundary points
    pred_points = np.argwhere(pred[0, 0] > 0)
    target_points = np.argwhere(target[0, 0] > 0)
    
    if len(pred_points) == 0 or len(target_points) == 0:
        return float('inf')
    
    # Scale by spacing
    pred_points = pred_points * spacing
    target_points = target_points * spacing
    
    # Compute directed Hausdorff distances
    hd1 = directed_hausdorff(pred_points, target_points)[0]
    hd2 = directed_hausdorff(target_points, pred_points)[0]
    
    # Return 95th percentile
    hd95 = np.percentile([hd1, hd2], 95)
    
    return hd95


def save_checkpoint(model, optimizer, epoch, save_path):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    

def load_checkpoint(model, optimizer, load_path):
    """Load training checkpoint"""
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']
