import torch, torch.nn as nn, torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    # L = w_bce * BCE + w_dice * Dice, computed separately for the OD and OC channels, then averaged.
    # Dice is computed per image, then averaged over the batch. Cups are small and vary a lot in
    # size, so batch-level Dice would be dominated by the images with large cups.
    def __init__(self, w_bce=0.5, w_dice=0.5, eps=1e-6):
        super().__init__()
        self.w_bce, self.w_dice, self.eps = w_bce, w_dice, eps

    def forward(self, logits, target):
        logits = logits.float()                                   # keep the loss in fp32 under AMP
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=(2, 3))  # (B,C)
        p = torch.sigmoid(logits)
        inter = (p * target).sum(dim=(2, 3))
        denom = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1 - (2 * inter + self.eps) / (denom + self.eps)                                   # (B,C)
        per_channel = (self.w_bce * bce + self.w_dice * dice_loss).mean(dim=0)                        # (C,)
        return per_channel.mean(), per_channel.detach()
