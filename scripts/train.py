"""Train the joint optic-disc/optic-cup segmentation model."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.glaucoma_seg.data import load_frames
from src.glaucoma_seg.transforms import get_transforms
from src.glaucoma_seg.dataset import make_loaders
from src.glaucoma_seg.model import build_model
from src.glaucoma_seg.losses import BCEDiceLoss
from src.glaucoma_seg.train_engine import fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from latest checkpoint")
    args = parser.parse_args()
    cfg_path = (ROOT / args.config).resolve()
    cfg = json.loads(cfg_path.read_text())
    data_cfg, train_cfg = cfg["data"], cfg["train"].copy()
    train_cfg["scheme"] = data_cfg["primary_scheme"]
    train_cfg["img_size"] = data_cfg["img_size"]

    seed = int(train_cfg["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu": print("Warning: CPU training will be substantially slower.")

    processed = (ROOT / args.processed_dir).resolve()
    frames = load_frames(processed, train_cfg["scheme"])
    print({name: len(frame) for name, frame in frames.items()}, "| split:", train_cfg["scheme"])
    train_tf, eval_tf = get_transforms(data_cfg)
    train_loader, val_loader, _ = make_loaders(
        frames, train_tf, eval_tf,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg.get("num_workers", 2),
        seed=seed,
    )

    encoder_weights = "imagenet" if train_cfg.get("use_pretrained_encoder", True) else None
    model = build_model(encoder_weights).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=train_cfg["sched_factor"],
        patience=train_cfg["sched_patience"], min_lr=train_cfg["min_lr"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    out = (ROOT / args.output_dir).resolve()
    state = fit(
        model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, train_cfg,
        ckpt_dir=out / "checkpoints", metrics_dir=out / "metrics", device=device,
        resume=not args.no_resume,
    )
    print(f"\nBest epoch: {state['best_epoch']} | best mean validation Dice: {state['best']:.4f}")

if __name__ == "__main__":
    main()
