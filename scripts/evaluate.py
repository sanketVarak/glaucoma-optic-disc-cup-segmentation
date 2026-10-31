"""Evaluate a trained checkpoint on the untouched test partition."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.glaucoma_seg.data import load_frames
from src.glaucoma_seg.transforms import get_transforms
from src.glaucoma_seg.model import build_model
from src.glaucoma_seg.evaluate import evaluate_frame, summarize, by_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--output-dir", default="outputs/metrics")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    scheme = cfg["data"]["primary_scheme"]
    frames = load_frames((ROOT / args.processed_dir).resolve(), scheme)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = (ROOT / args.checkpoint).resolve()
    if not ckpt_path.exists(): raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(None).to(device)
    model.load_state_dict(checkpoint["model"])
    _, eval_tf = get_transforms(cfg["data"])
    ev = cfg["evaluation"]
    results = evaluate_frame(
        model, frames["test"], eval_tf, device,
        mode=ev["mode"], threshold=ev["threshold"],
        enforce_containment=ev.get("enforce_containment", True),
    )
    out = (ROOT / args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "test_per_image.csv")
    summary = summarize(results)
    summary.to_csv(out / "test_summary.csv", index=False)
    by_group(results, ["source", "label"]).to_csv(out / "test_by_source_label.csv")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Test images: {len(results)}")
    print(summary.round(4).to_string(index=False))
    print("\nMean Dice:")
    print(results[["od_dice", "oc_dice"]].mean().round(4).to_string())
    print(f"\nLocal evaluation files saved to {out}; these CSVs are git-ignored.")

if __name__ == "__main__":
    main()
