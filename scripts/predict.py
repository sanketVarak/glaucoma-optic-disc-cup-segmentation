"""Run optic-disc and optic-cup segmentation on one fundus image."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.glaucoma_seg.transforms import get_transforms
from src.glaucoma_seg.model import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to an input fundus image")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output-dir", default="outputs/predictions")
    args = parser.parse_args()
    image_path = Path(args.image).resolve()
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None: raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    cfg = json.loads((ROOT / args.config).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load((ROOT / args.checkpoint).resolve(), map_location=device, weights_only=False)
    model = build_model(None).to(device); model.load_state_dict(checkpoint["model"]); model.eval()
    _, eval_tf = get_transforms(cfg["data"])
    tensor = eval_tf(image=image)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor).float()
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        masks = (torch.sigmoid(logits)[0] >= cfg["evaluation"]["threshold"]).cpu().numpy()
    masks[1] &= masks[0]  # enforce cup-inside-disc constraint
    out = (ROOT / args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    cv2.imwrite(str(out / f"{stem}_disc.png"), masks[0].astype(np.uint8) * 255)
    cv2.imwrite(str(out / f"{stem}_cup.png"), masks[1].astype(np.uint8) * 255)
    print(f"Saved masks to {out}")

if __name__ == "__main__":
    main()
