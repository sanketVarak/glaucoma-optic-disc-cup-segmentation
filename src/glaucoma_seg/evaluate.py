import cv2, numpy as np, pandas as pd, torch
import torch.nn.functional as F

CH = ("od", "oc")
METRICS = ("dice", "iou", "sensitivity", "precision", "specificity")

def confusion_metrics(pred, gt):
    # pred, gt: boolean (H, W) arrays. All metrics come from the same four counts.
    tp = int((pred & gt).sum());  fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum()); tn = int((~pred & ~gt).sum())
    p, g, nan = tp + fp, tp + fn, float("nan")
    return dict(
        dice=1.0 if p + g == 0 else 2 * tp / (p + g),          # both empty -> perfect agreement
        iou=1.0 if p + g == 0 else tp / (p + g - tp),
        sensitivity=tp / g if g > 0 else nan,                  # = recall
        precision=tp / p if p > 0 else nan,                    # undefined if nothing was predicted
        specificity=tn / (tn + fp) if tn + fp > 0 else nan,
        pred_px=p, gt_px=g)

def _read(path, flag):
    a = cv2.imread(str(path), flag)
    if a is None:
        raise FileNotFoundError(path)
    return a

@torch.no_grad()
def evaluate_frame(model, df, eval_tf, device, mode="original", threshold=0.5, enforce_containment=True):
    # df: index = stem, columns img_path, disc_path, cup_path, label, source
    # mode "original": logits are upsampled to each image's native size and compared with the original masks
    # mode "resized" : compared at the model's input size (masks resized with nearest-neighbour)
    if mode not in ("original", "resized"):
        raise ValueError(mode)
    model.eval()
    rows = []
    for stem, r in df.iterrows():
        img = cv2.cvtColor(_read(r["img_path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        gts = [_read(r["disc_path"], cv2.IMREAD_GRAYSCALE) > 0, _read(r["cup_path"], cv2.IMREAD_GRAYSCALE) > 0]
        H, W = img.shape[:2]
        x = eval_tf(image=img)["image"].unsqueeze(0).to(device)
        logits = model(x).float()                               # (1, 2, S, S); fp32 for evaluation
        if mode == "original":
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        else:
            S = logits.shape[-1]
            gts = [cv2.resize(g.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST) > 0 for g in gts]
        pred = (torch.sigmoid(logits)[0] >= threshold).cpu().numpy()
        od, oc = pred[0], pred[1]
        viol = float((oc & ~od).sum() / max(int(oc.sum()), 1))  # raw cup pixels outside the predicted disc
        if enforce_containment:
            oc = oc & od
        row = dict(stem=stem, label=r["label"], source=r["source"], H=H, W=W, cup_outside_disc_raw=viol)
        for ch, p, g in zip(CH, (od, oc), gts):
            for k, v in confusion_metrics(p, g).items():
                row[f"{ch}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows).set_index("stem")

def summarize(res):
    out = []
    for ch in CH:
        for m in METRICS:
            v = res[f"{ch}_{m}"]
            out.append(dict(structure=ch.upper(), metric=m, mean=v.mean(), std=v.std(ddof=1),
                            median=v.median(), min=v.min(), n_undefined=int(v.isna().sum())))
    return pd.DataFrame(out)

def by_group(res, keys):
    return res.groupby(keys).agg(
        n=("od_dice", "size"),
        od_dice_mean=("od_dice", "mean"), od_dice_std=("od_dice", "std"),
        oc_dice_mean=("oc_dice", "mean"), oc_dice_std=("oc_dice", "std"),
        od_iou_mean=("od_iou", "mean"), oc_iou_mean=("oc_iou", "mean"))

def bootstrap_ci(x, n=2000, seed=42, alpha=0.05):
    # percentile bootstrap over images; ignores any patient-level correlation (IDs are unknown)
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    means = np.random.default_rng(seed).choice(x, (n, len(x)), replace=True).mean(1)
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))
