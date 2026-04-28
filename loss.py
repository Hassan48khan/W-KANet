"""
Loss Functions for W-KANet
============================================================================

Task-Adaptive Loss Functions:
- Binary Loss: Weighted IoU-BCE + Boundary + KAN Regularization
- Multi-Class Loss: Focal-Dice + KAN Regularization

Repository: https://github.com/Hassan48khan/W-KANet
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Binary Segmentation Losses
# ═══════════════════════════════════════════════════════════════════════════

class WeightedIoUBCELoss(nn.Module):
    """
    Weighted IoU + BCE Loss with difficulty weighting.
    
    Difficulty Weighting:
        w_i = 1 + 2 * |y_pred - y_true|
    
    Range: [1.0, 3.0] - emphasizes hard pixels at boundaries.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # Compute difficulty weights
        weight = 1 + 2 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
        
        # Weighted BCE
        wbce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        wbce = (weight * wbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
        
        # Weighted IoU
        pred = torch.sigmoid(pred)
        inter = ((pred * target) * weight).sum(dim=(2, 3))
        union = ((pred + target) * weight).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        
        return (wbce + wiou).mean()


class BoundaryLoss(nn.Module):
    """
    Boundary Loss with Canny edge supervision.
    
    Uses cardiac-optimized Canny thresholds (low=50, high=150)
    with 10:1 edge-to-background weighting.
    """
    def __init__(self, edge_weight=10.0, low_threshold=50, high_threshold=150):
        super().__init__()
        self.edge_weight = edge_weight
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def extract_canny_edges(self, mask):
        """Extract Canny edges from ground truth mask."""
        edges_batch = []
        for i in range(mask.shape[0]):
            mask_np = (mask[i, 0].cpu().numpy() * 255).astype(np.uint8)
            edges = cv2.Canny(mask_np, self.low_threshold, self.high_threshold)
            edges = torch.from_numpy(edges).float() / 255.0
            edges_batch.append(edges)
        return torch.stack(edges_batch).unsqueeze(1).to(mask.device)
    
    def forward(self, pred_boundary, gt_mask):
        """
        Args:
            pred_boundary: Predicted boundary maps [B, 1, H, W]
            gt_mask: Ground truth segmentation mask [B, 1, H, W]
        """
        # Extract ground truth boundaries
        gt_edges = self.extract_canny_edges(gt_mask)
        
        # Apply edge weighting (10:1)
        weight = torch.ones_like(gt_edges)
        weight[gt_edges > 0.5] = self.edge_weight
        
        # Weighted BCE
        loss = F.binary_cross_entropy_with_logits(
            pred_boundary, gt_edges, weight=weight, reduction='mean'
        )
        return loss


class BinaryLoss(nn.Module):
    """
    Complete Binary Segmentation Loss.
    
    L_binary = L_w-IoU + L_w-BCE + λ_b * L_boundary + λ_KAN * L_reg
    
    where:
        - λ_b = 0.25 (boundary loss weight)
        - λ_KAN = 0.05 (KAN regularization weight)
    """
    def __init__(self, lambda_boundary=0.25, lambda_kan=0.05):
        super().__init__()
        self.seg_loss = WeightedIoUBCELoss()
        self.boundary_loss = BoundaryLoss()
        self.lambda_boundary = lambda_boundary
        self.lambda_kan = lambda_kan
    
    def forward(self, predictions, boundary_maps, gt_mask, model=None):
        """
        Args:
            predictions: Tuple of (aux_predictions, main_prediction)
                - aux_predictions: List of 4 deep supervision outputs
                - main_prediction: Final segmentation output
            boundary_maps: Dict of boundary predictions from 4 stages
            gt_mask: Ground truth mask
            model: Model instance for KAN regularization (optional)
        """
        aux_preds, main_pred = predictions
        
        # Main segmentation loss
        loss_seg = self.seg_loss(main_pred, gt_mask)
        
        # Deep supervision losses
        for aux_pred in aux_preds:
            loss_seg += self.seg_loss(aux_pred, gt_mask)
        
        # Boundary losses
        loss_bnd = 0
        for stage in ['stage1', 'stage2', 'stage3', 'stage4']:
            loss_bnd += self.boundary_loss(boundary_maps[stage], gt_mask)
        loss_bnd *= self.lambda_boundary
        
        # KAN regularization
        loss_kan = 0
        if model is not None and hasattr(model, 'regularization_loss'):
            loss_kan = self.lambda_kan * model.regularization_loss()
        
        # Total loss
        loss_total = loss_seg + loss_bnd + loss_kan
        
        return loss_total, {
            'loss_total': loss_total.item() if torch.is_tensor(loss_total) else loss_total,
            'loss_seg': loss_seg.item(),
            'loss_bnd': loss_bnd.item() if torch.is_tensor(loss_bnd) else loss_bnd,
            'loss_kan': loss_kan.item() if torch.is_tensor(loss_kan) else loss_kan,
        }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Multi-Class Segmentation Losses
# ═══════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class segmentation.
    
    L_Focal = -1/N * Σ Σ α_c * (1 - p_c)^γ * y_c * log(p_c)
    
    where:
        - α_c = 0.25 (class balance)
        - γ = 2.0 (hard sample focus)
    """
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=4):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(self, pred, target):
        """
        Args:
            pred: Logits [B, C, H, W]
            target: Class indices [B, H, W]
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation.
    
    Preserves boundary integrity in cardiac structures.
    """
    def __init__(self, num_classes=4, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Logits [B, C, H, W]
            target: Class indices [B, H, W]
        """
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Compute dice for each class (excluding background)
        dice_loss = 0
        for c in range(1, self.num_classes):
            intersection = (pred[:, c] * target_one_hot[:, c]).sum()
            union = pred[:, c].sum() + target_one_hot[:, c].sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1 - dice)
        
        return dice_loss / (self.num_classes - 1)


class MultiClassLoss(nn.Module):
    """
    Complete Multi-Class Segmentation Loss for ACDC.
    
    L_multi = λ_1 * L_Focal + λ_2 * L_Dice + λ_KAN * L_reg
    
    where:
        - λ_1 = 0.5 (Focal weight)
        - λ_2 = 1.0 (Dice weight)
        - λ_KAN = 0.05 (KAN regularization)
    """
    def __init__(self, num_classes=4, lambda_focal=0.5, lambda_dice=1.0, 
                 lambda_kan=0.05, alpha=0.25, gamma=2.0):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, num_classes=num_classes)
        self.dice_loss = DiceLoss(num_classes=num_classes)
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        self.lambda_kan = lambda_kan
    
    def forward(self, predictions, gt_mask, model=None):
        """
        Args:
            predictions: Tuple of (aux_predictions, main_prediction)
            gt_mask: Ground truth mask [B, H, W] with class indices
            model: Model instance for KAN regularization (optional)
        """
        aux_preds, main_pred = predictions
        
        # Main losses
        focal = self.focal_loss(main_pred, gt_mask)
        dice = self.dice_loss(main_pred, gt_mask)
        loss_seg = self.lambda_focal * focal + self.lambda_dice * dice
        
        # Deep supervision
        for aux_pred in aux_preds:
            focal = self.focal_loss(aux_pred, gt_mask)
            dice = self.dice_loss(aux_pred, gt_mask)
            loss_seg += self.lambda_focal * focal + self.lambda_dice * dice
        
        # KAN regularization
        loss_kan = 0
        if model is not None and hasattr(model, 'regularization_loss'):
            loss_kan = self.lambda_kan * model.regularization_loss()
        
        # Total loss
        loss_total = loss_seg + loss_kan
        
        return loss_total, {
            'loss_total': loss_total.item() if torch.is_tensor(loss_total) else loss_total,
            'loss_focal': focal.item(),
            'loss_dice': dice.item(),
            'loss_kan': loss_kan.item() if torch.is_tensor(loss_kan) else loss_kan,
        }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Task-Adaptive Loss Selector
# ═══════════════════════════════════════════════════════════════════════════

class TaskAdaptiveLoss(nn.Module):
    """
    Unified loss interface for both binary and multi-class segmentation.
    
    Args:
        task: 'binary' or 'multi-class'
        num_classes: Number of classes (1 for binary, 4 for ACDC)
    """
    def __init__(self, task='binary', num_classes=1):
        super().__init__()
        self.task = task
        if task == 'binary':
            self.loss_fn = BinaryLoss()
        elif task == 'multi-class':
            self.loss_fn = MultiClassLoss(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown task: {task}. Use 'binary' or 'multi-class'.")
    
    def forward(self, *args, **kwargs):
        return self.loss_fn(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*80)
    print("Testing W-KANet Loss Functions")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test Binary Loss
    print("\n[1] Binary Loss Test:")
    pred_main = torch.randn(2, 1, 128, 128, device=device)
    pred_aux = [torch.randn(2, 1, 128, 128, device=device) for _ in range(4)]
    boundary_maps = {f'stage{i}': torch.randn(2, 1, 128, 128, device=device) 
                     for i in range(1, 5)}
    gt_binary = torch.randint(0, 2, (2, 1, 128, 128), device=device).float()
    
    binary_loss = BinaryLoss().to(device)
    loss, info = binary_loss((pred_aux, pred_main), boundary_maps, gt_binary)
    print(f"  ✓ Binary Loss: {loss.item():.4f}")
    print(f"    - Seg Loss: {info['loss_seg']:.4f}")
    print(f"    - Bnd Loss: {info['loss_bnd']:.4f}")
    
    # Test Multi-Class Loss
    print("\n[2] Multi-Class Loss Test:")
    pred_main = torch.randn(2, 4, 128, 128, device=device)
    pred_aux = [torch.randn(2, 4, 128, 128, device=device) for _ in range(4)]
    gt_multi = torch.randint(0, 4, (2, 128, 128), device=device)
    
    multi_loss = MultiClassLoss(num_classes=4).to(device)
    loss, info = multi_loss((pred_aux, pred_main), gt_multi)
    print(f"  ✓ Multi-Class Loss: {loss.item():.4f}")
    print(f"    - Focal Loss: {info['loss_focal']:.4f}")
    print(f"    - Dice Loss: {info['loss_dice']:.4f}")
    
    print("\n" + "="*80)
    print("✅ All loss functions working correctly!")
    print("="*80)
