import torch

@torch.no_grad()
def dice_iou(pred, target, eps=1e-7):
    # pred, target: (B, C, H, W) binary. Returns per-image, per-channel Dice and IoU, each (B, C).
    # If prediction and ground truth are both empty, agreement is perfect by convention (score 1).
    pred, target = pred.float(), target.float()
    tp = (pred * target).sum((2, 3))
    p, g = pred.sum((2, 3)), target.sum((2, 3))
    empty = (p + g) == 0
    dice = torch.where(empty, torch.ones_like(tp), 2 * tp / (p + g).clamp(min=eps))
    iou = torch.where(empty, torch.ones_like(tp), tp / (p + g - tp).clamp(min=eps))
    return dice, iou
