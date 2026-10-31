import torch
import segmentation_models_pytorch as smp

def build_model(encoder_weights="imagenet"):
    # 2 output channels, ch0 = optic disc and ch1 = optic cup. activation=None, so the model returns
    # raw logits: BCE-with-logits is numerically stabler in training, and sigmoid is applied at inference.
    return smp.Unet(encoder_name="resnet34", encoder_weights=encoder_weights,
                    in_channels=3, classes=2, activation=None)

def containment_violation(masks):
    # fraction of predicted cup pixels lying outside the predicted disc, per sample.
    # masks: (B, 2, H, W) binary, ch0 = disc, ch1 = cup
    cup, disc = masks[:, 1] > 0, masks[:, 0] > 0
    outside = (cup & ~disc).flatten(1).sum(1).float()
    return outside / cup.flatten(1).sum(1).clamp(min=1).float()

@torch.no_grad()
def predict_masks(model, x, threshold=0.5, enforce_containment=True):
    # sigmoid -> threshold. The cup is anatomically inside the disc, so optionally cup = cup AND disc.
    model.eval()
    probs = torch.sigmoid(model(x).float())
    masks = (probs >= threshold)
    if enforce_containment:
        masks[:, 1] = masks[:, 1] & masks[:, 0]
    return probs, masks.float()
