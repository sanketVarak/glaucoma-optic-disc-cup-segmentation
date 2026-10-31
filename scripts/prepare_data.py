"""Create local manifest/split CSVs from the RIM-ONE DL images and masks."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.glaucoma_seg.data import build_manifest, make_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", default="data/raw/RIM_ONE_DL_images")
    parser.add_argument("--masks-root", default="data/raw/RIM_ONE_DL_reference_segmentations")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    images_root = (ROOT / args.images_root).resolve()
    masks_root = (ROOT / args.masks_root).resolve()
    out = (ROOT / args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(images_root, masks_root)
    manifest.to_csv(out / "manifest.csv")
    print(f"Indexed {len(manifest)} unique images.")
    print("Class counts:")
    print(manifest["label"].value_counts().to_string())
    print("Source counts:")
    print(manifest["source"].value_counts().to_string())

    for scheme in ("hospital", "random"):
        splits = make_split(manifest, scheme, args.val_frac, args.seed)
        splits.to_csv(out / f"splits_{scheme}.csv")
        print(f"\n{scheme} split counts:")
        print(splits.groupby(["subset", "label"]).size().to_string())
    print(f"\nPrepared files saved under: {out}")

if __name__ == "__main__":
    main()
