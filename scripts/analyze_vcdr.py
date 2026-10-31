"""Analyse VCDR agreement and exploratory label discrimination on the test set."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.glaucoma_seg.data import load_frames
from src.glaucoma_seg.transforms import get_transforms
from src.glaucoma_seg.model import build_model
from src.glaucoma_seg.vcdr import largest_component, vcdr


def agreement(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Return correlation, MAE, bias and 95% Bland-Altman limits."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    err = y - x
    sd = err.std(ddof=1)
    return {
        "pearson_r": float(pearsonr(x, y).statistic),
        "spearman_r": float(spearmanr(x, y).statistic),
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
        "loa_low": float(err.mean() - 1.96 * sd),
        "loa_high": float(err.mean() + 1.96 * sd),
    }


def bootstrap_auc_ci(score: np.ndarray, labels: np.ndarray, n: int = 1000,
                     seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        if np.unique(labels[idx]).size == 2:
            values.append(roc_auc_score(labels[idx], score[idx]))
    return tuple(np.quantile(values, [0.025, 0.975]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    cfg = json.loads((ROOT / args.config).read_text())
    scheme = cfg["data"]["primary_scheme"]
    frames = load_frames((ROOT / args.processed_dir).resolve(), scheme)
    test_df = frames["test"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = (ROOT / args.checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(None).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    _, eval_tf = get_transforms(cfg["data"])
    threshold = float(cfg["evaluation"]["threshold"])

    rows = []
    with torch.no_grad():
        for stem, row in test_df.iterrows():
            image_bgr = cv2.imread(str(row["img_path"]), cv2.IMREAD_COLOR)
            disc_gt = cv2.imread(str(row["disc_path"]), cv2.IMREAD_GRAYSCALE)
            cup_gt = cv2.imread(str(row["cup_path"]), cv2.IMREAD_GRAYSCALE)
            if image_bgr is None or disc_gt is None or cup_gt is None:
                raise FileNotFoundError(f"Could not load image or masks for {stem}")
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            tensor = eval_tf(image=image)["image"].unsqueeze(0).to(device)
            logits = model(tensor).float()
            logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
            pred = (torch.sigmoid(logits)[0] >= threshold).cpu().numpy()
            pred[1] &= pred[0]  # enforce cup-inside-disc containment

            gt_disc = disc_gt > 0
            gt_cup = cup_gt > 0
            pred_disc = pred[0].astype(bool)
            pred_cup = pred[1].astype(bool)
            pred_disc_lcc = largest_component(pred_disc)
            pred_cup_lcc = largest_component(pred_cup) & pred_disc_lcc
            rows.append({
                "stem": stem,
                "label": row["label"],
                "source": row["source"],
                "vcdr_gt": vcdr(gt_disc, gt_cup),
                "vcdr_pred": vcdr(pred_disc, pred_cup),
                "vcdr_pred_lcc": vcdr(pred_disc_lcc, pred_cup_lcc),
                "cup_gt_fraction": float(gt_cup.mean()),
                "height": height,
                "width": width,
            })

    results = pd.DataFrame(rows).set_index("stem")
    results["vcdr_error"] = results["vcdr_pred"] - results["vcdr_gt"]
    results["vcdr_lcc_error"] = results["vcdr_pred_lcc"] - results["vcdr_gt"]
    labels = (results["label"].to_numpy() == "glaucoma").astype(int)
    if np.unique(labels).size != 2:
        raise ValueError("The test partition must contain both normal and glaucoma labels for ROC-AUC.")

    out = (ROOT / args.output_dir).resolve()
    metrics_dir, graphs_dir = out / "metrics", out / "graphs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(metrics_dir / "test_vcdr.csv")

    primary = agreement(results["vcdr_gt"], results["vcdr_pred"])
    lcc = agreement(results["vcdr_gt"], results["vcdr_pred_lcc"])
    auc_gt = roc_auc_score(labels, results["vcdr_gt"])
    auc_pred = roc_auc_score(labels, results["vcdr_pred"])
    auc_gt_ci = bootstrap_auc_ci(results["vcdr_gt"].to_numpy(), labels)
    auc_pred_ci = bootstrap_auc_ci(results["vcdr_pred"].to_numpy(), labels)

    print(f"Test images: {len(results)}")
    print("\nPredicted VCDR vs ground truth:")
    print(pd.Series(primary).round(3).to_string())
    print("\nLargest-component VCDR vs ground truth:")
    print(pd.Series(lcc).round(3).to_string())
    print(f"\nVCDR-only ROC-AUC (ground-truth VCDR): {auc_gt:.3f} [{auc_gt_ci[0]:.3f}, {auc_gt_ci[1]:.3f}]")
    print(f"VCDR-only ROC-AUC (predicted VCDR):   {auc_pred:.3f} [{auc_pred_ci[0]:.3f}, {auc_pred_ci[1]:.3f}]")

    colors = np.where(labels == 1, "tab:red", "tab:blue")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].scatter(results["vcdr_gt"], results["vcdr_pred"], c=colors, s=16, alpha=0.7)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="Ground-truth VCDR", ylabel="Predicted VCDR", xlim=(0, 1), ylim=(0, 1),
                aspect="equal", title=f"Agreement (r = {primary['pearson_r']:.2f}, MAE = {primary['mae']:.3f})")

    means = (results["vcdr_gt"] + results["vcdr_pred"]) / 2
    errors = results["vcdr_error"]
    axes[1].scatter(means, errors, c=colors, s=16, alpha=0.7)
    axes[1].axhline(0, c="gray", lw=1)
    for value, style in ((primary["bias"], "-"), (primary["loa_low"], ":"), (primary["loa_high"], ":")):
        axes[1].axhline(value, c="k", ls=style, lw=1)
    axes[1].set(xlabel="Mean of predicted and ground-truth VCDR", ylabel="Predicted − ground truth",
                title=f"Bland–Altman (bias {primary['bias']:+.3f}, limits {primary['loa_low']:+.3f} to {primary['loa_high']:+.3f})")

    for key, name in (("vcdr_gt", "Ground-truth VCDR"), ("vcdr_pred", "Predicted VCDR")):
        fpr, tpr, _ = roc_curve(labels, results[key])
        auc = roc_auc_score(labels, results[key])
        axes[2].plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
    axes[2].plot([0, 1], [0, 1], "k:", lw=1)
    axes[2].set(xlabel="1 − specificity", ylabel="Sensitivity", title="Glaucoma vs normal from VCDR alone")
    axes[2].legend(frameon=False)
    axes[0].scatter([], [], c="tab:blue", label="Normal")
    axes[0].scatter([], [], c="tab:red", label="Glaucoma")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(graphs_dir / "vcdr_agreement.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved VCDR table and plot under: {out}")


if __name__ == "__main__":
    main()
