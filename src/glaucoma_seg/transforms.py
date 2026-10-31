import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

CFG = dict(seed=42, img_size=512, val_frac=0.15, primary_scheme="hospital",
           mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

def _resize(s):
    # bilinear for the image, nearest-neighbour for masks so they stay strictly {0, 1}
    return A.Resize(s, s, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST)

def get_transforms(cfg=CFG):
    s = cfg["img_size"]          # images are square, so a plain resize keeps the aspect ratio
    tail = lambda: [A.Normalize(mean=cfg["mean"], std=cfg["std"]), ToTensorV2()]
    train_tf = A.Compose([
        _resize(s),
        A.HorizontalFlip(p=0.5),   # no vertical flip: up/down is anatomically meaningful (VCDR)
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        *tail()])
    eval_tf = A.Compose([_resize(s), *tail()])
    return train_tf, eval_tf
