import json, os, random, time
from pathlib import Path
import numpy as np
import torch
from .metrics import dice_iou

def _save(obj, path):
    # write to a temp file, then rename: a crash mid-write can't corrupt the existing checkpoint
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

def _rng_state():
    return dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

def _set_rng_state(s):
    random.setstate(s["py"]); np.random.set_state(s["np"]); torch.set_rng_state(s["torch"].cpu())
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    use_amp = device == "cuda"
    tot, n, ch = 0.0, 0, torch.zeros(2, device=device)
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        loss, per_ch = criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        b = x.size(0); tot += loss.item() * b; ch += per_ch * b; n += b
    return tot / n, (ch / n).tolist()

@torch.no_grad()
def validate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    use_amp = device == "cuda"
    tot, n, D, I = 0.0, 0, [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        loss, _ = criterion(logits, y)
        tot += loss.item() * x.size(0); n += x.size(0)
        pred = torch.sigmoid(logits.float()) >= threshold
        pred[:, 1] = pred[:, 1] & pred[:, 0]               # cup must lie inside the predicted disc
        d, i = dice_iou(pred, y)
        D.append(d.cpu()); I.append(i.cpu())
    D, I = torch.cat(D), torch.cat(I)                      # (N, 2): per image, per channel
    return dict(loss=tot / n, dice_od=D[:, 0].mean().item(), dice_oc=D[:, 1].mean().item(),
                iou_od=I[:, 0].mean().item(), iou_oc=I[:, 1].mean().item(), dice_mean=D.mean().item())

def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, cfg,
        ckpt_dir, metrics_dir, device, resume=True):
    ckpt_dir, metrics_dir = Path(ckpt_dir), Path(metrics_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True); metrics_dir.mkdir(parents=True, exist_ok=True)
    latest, best = ckpt_dir / "latest.pt", ckpt_dir / "best.pt"
    state = dict(epoch=0, best=-1.0, best_epoch=0, bad=0, history=[])

    if resume and latest.exists():
        # weights_only=False because the file holds RNG states (numpy/python objects): load only your own files
        ck = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        state = ck["state"]; _set_rng_state(ck["rng"])
        if ck["cfg"] != cfg:
            print("WARNING: config differs from the one stored in the checkpoint")
        print(f"resumed after epoch {state['epoch']} (best Dice {state['best']:.4f} at epoch {state['best_epoch']})")
    else:
        print("starting fresh")

    for epoch in range(state["epoch"] + 1, cfg["max_epochs"] + 1):
        if state["bad"] >= cfg["early_stop_patience"]:
            print(f"early stopping: no improvement for {state['bad']} epochs"); break
        t0 = time.time()
        # the shuffle order and the augmentation seeds both derive from this generator,
        # so seeding it per epoch makes every epoch reproducible and resumable
        train_loader.generator.manual_seed(cfg["seed"] + epoch)
        tr_loss, tr_ch = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val = validate(model, val_loader, criterion, device, cfg["threshold"])
        lr = optimizer.param_groups[0]["lr"]               # the lr this epoch actually used
        scheduler.step(val["dice_mean"])

        improved = val["dice_mean"] > state["best"] + cfg["min_delta"]
        if improved:
            state.update(best=val["dice_mean"], best_epoch=epoch, bad=0)
            _save(dict(model=model.state_dict(), epoch=epoch, val=val, cfg=cfg), best)
        else:
            state["bad"] += 1
        state["epoch"] = epoch
        state["history"].append(dict(epoch=epoch, lr=lr, train_loss=tr_loss, train_loss_od=tr_ch[0],
                                     train_loss_oc=tr_ch[1], seconds=time.time() - t0,
                                     **{f"val_{k}": v for k, v in val.items()}))
        _save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                   scaler=scaler.state_dict(), state=state, rng=_rng_state(), cfg=cfg), latest)
        (metrics_dir / "history.json").write_text(json.dumps(state["history"], indent=1))

        tag = "*best*" if improved else f"no gain {state['bad']}/{cfg['early_stop_patience']}"
        print(f"ep {epoch:2d} | lr {lr:.1e} | train {tr_loss:.4f} | val loss {val['loss']:.4f} | "
              f"Dice OD {val['dice_od']:.4f} OC {val['dice_oc']:.4f} (mean {val['dice_mean']:.4f}) | "
              f"{tag} | {time.time() - t0:.0f}s")
    return state
