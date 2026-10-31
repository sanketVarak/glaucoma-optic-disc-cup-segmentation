"""Dataset indexing helpers for the RIM-ONE DL folder layout."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

SCHEME_DIRS = {
    "partitioned_by_hospital": "hospital",
    "partitioned_randomly": "random",
}
MASK_PATTERN = re.compile(r"^(?P<stem>.+)-1-(?P<part>Disc|Cup)-T\.png$", re.IGNORECASE)


def build_manifest(images_root: str | Path, masks_root: str | Path) -> pd.DataFrame:
    """Index both official image partitions and pair each image with its Disc/Cup masks.

    Expected image paths include:
      partitioned_by_hospital/{training_set,test_set}/{glaucoma,normal}/*.png
      partitioned_randomly/{training_set,test_set}/{glaucoma,normal}/*.png
    Mask filenames follow ``<image_stem>-1-Disc-T.png`` and ``...-Cup-T.png``.
    """
    images_root, masks_root = Path(images_root), Path(masks_root)
    if not images_root.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_root}")
    if not masks_root.is_dir():
        raise FileNotFoundError(f"Masks directory not found: {masks_root}")

    image_rows = []
    for path in sorted(images_root.rglob("*.png")):
        parts = path.relative_to(images_root).parts
        hit = next((i for i, part in enumerate(parts) if part in SCHEME_DIRS), None)
        if hit is None or len(parts) < hit + 4:
            continue
        scheme = SCHEME_DIRS[parts[hit]]
        split_name, label = parts[hit + 1], parts[hit + 2].lower()
        if split_name not in {"training_set", "test_set"}:
            continue
        if label not in {"normal", "glaucoma"}:
            continue
        image_rows.append({
            "stem": path.stem,
            "scheme": scheme,
            "split": split_name,
            "label": label,
            "path": str(path.resolve()),
        })
    if not image_rows:
        raise RuntimeError("No PNG images found in the expected RIM-ONE DL partition folders.")

    image_df = pd.DataFrame(image_rows)
    if image_df.duplicated(["stem", "scheme"]).any():
        dupes = image_df.loc[image_df.duplicated(["stem", "scheme"], keep=False), "stem"].tolist()
        raise ValueError(f"Duplicate image stems within a partition: {dupes[:10]}")
    labels_per_stem = image_df.groupby("stem")["label"].nunique()
    if (labels_per_stem > 1).any():
        raise ValueError("At least one image stem has conflicting normal/glaucoma labels.")

    split_wide = image_df.pivot(index="stem", columns="scheme", values="split").add_prefix("split_")
    image_wide = image_df.pivot(index="stem", columns="scheme", values="path").add_prefix("img_")
    labels = image_df.groupby("stem")["label"].first().rename("label")

    mask_rows = []
    for path in sorted(masks_root.rglob("*.png")):
        match = MASK_PATTERN.match(path.name)
        if not match:
            continue
        mask_rows.append({
            "stem": match.group("stem"),
            "part": match.group("part").lower(),
            "mask_label": path.parent.name.lower(),
            "path": str(path.resolve()),
        })
    if not mask_rows:
        raise RuntimeError("No masks matching '*-1-Disc-T.png' / '*-1-Cup-T.png' were found.")
    mask_df = pd.DataFrame(mask_rows)
    if mask_df.duplicated(["stem", "part"]).any():
        raise ValueError("Duplicate mask files found for an image/structure pair.")
    mask_wide = mask_df.pivot(index="stem", columns="part", values="path").rename(
        columns={"disc": "disc_path", "cup": "cup_path"}
    )
    mask_labels = mask_df.groupby("stem")["mask_label"].first().rename("mask_label")

    manifest = split_wide.join(image_wide).join(labels).join(mask_wide).join(mask_labels)
    required = ["split_hospital", "split_random", "img_hospital", "img_random", "disc_path", "cup_path"]
    missing = [c for c in required if c not in manifest.columns or manifest[c].isna().any()]
    if missing:
        raise ValueError(f"Incomplete image/mask pairs or missing partition entries: {missing}")
    if not (manifest["label"] == manifest["mask_label"]).all():
        raise ValueError("Image class folder and mask class folder do not match for every sample.")

    manifest["source"] = manifest.index.to_series().str.split("_").str[0]
    widths, heights, disc_fracs, cup_fracs, cup_outside = [], [], [], [], []
    for row in manifest.itertuples():
        with Image.open(row.img_hospital) as im:
            width, height = im.size
        disc = np.asarray(Image.open(row.disc_path)) > 0
        cup = np.asarray(Image.open(row.cup_path)) > 0
        if disc.shape != (height, width) or cup.shape != (height, width):
            raise ValueError(f"Image/mask dimensions differ for {row.Index}.")
        widths.append(width); heights.append(height)
        disc_fracs.append(float(disc.mean())); cup_fracs.append(float(cup.mean()))
        cup_outside.append(int((cup & ~disc).sum()))
    manifest["img_w"] = widths
    manifest["img_h"] = heights
    manifest["disc_frac"] = disc_fracs
    manifest["cup_frac"] = cup_fracs
    manifest["cup_outside_disc_px"] = cup_outside
    manifest.index.name = "stem"
    return manifest.sort_index()


def make_split(manifest: pd.DataFrame, scheme: str = "hospital", val_frac: float = 0.15,
               seed: int = 42) -> pd.DataFrame:
    """Keep the supplied test partition fixed; stratify validation from training only."""
    if scheme not in {"hospital", "random"}:
        raise ValueError("scheme must be 'hospital' or 'random'")
    split_col = f"split_{scheme}"
    if split_col not in manifest:
        raise KeyError(f"Manifest has no {split_col} column")
    result = manifest[["label", "source", split_col]].copy()
    result["subset"] = result[split_col].map({"training_set": "train", "test_set": "test"})
    if result["subset"].isna().any():
        raise ValueError(f"Unexpected split labels in {split_col}: {result[split_col].unique()}")
    result["stratum"] = result["label"].astype(str) + "_" + result["source"].astype(str)
    train_ids = result.index[result["subset"] == "train"]
    _, val_ids = train_test_split(
        train_ids, test_size=val_frac, stratify=result.loc[train_ids, "stratum"], random_state=seed
    )
    result.loc[val_ids, "subset"] = "val"
    result = result.drop(columns=split_col)
    if result["subset"].isna().any():
        raise AssertionError("Split generation produced missing subsets.")
    return result


def load_frames(processed_dir: str | Path, scheme: str = "hospital"):
    """Load locally generated manifest/split tables into train/val/test DataFrames."""
    processed_dir = Path(processed_dir)
    manifest_path = processed_dir / "manifest.csv"
    split_path = processed_dir / f"splits_{scheme}.csv"
    if not manifest_path.exists() or not split_path.exists():
        raise FileNotFoundError(
            "Prepared CSVs are missing. Run scripts/prepare_data.py before training/evaluation."
        )
    manifest = pd.read_csv(manifest_path, index_col=0)
    split = pd.read_csv(split_path, index_col=0)
    img_col = f"img_{scheme}"
    if img_col not in manifest.columns:
        img_col = "img_hospital"
    columns = [img_col, "disc_path", "cup_path", "label", "source"]
    missing = [col for col in columns if col not in manifest.columns]
    if missing:
        raise KeyError(f"Manifest is missing required columns: {missing}")
    frames = {}
    for subset in ("train", "val", "test"):
        ids = split.index[split["subset"] == subset]
        frames[subset] = manifest.loc[ids, columns].rename(columns={img_col: "img_path"})
    return frames
