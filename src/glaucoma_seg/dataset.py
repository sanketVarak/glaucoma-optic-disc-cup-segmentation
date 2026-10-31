import random
import cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

NEED = ("img_path", "disc_path", "cup_path", "label", "source")

class RimOneDataset(Dataset):
    # df: one row per image, index = image stem, columns = NEED
    def __init__(self, df, transform):
        missing = [c for c in NEED if c not in df.columns]
        if missing:
            raise KeyError(f"missing columns: {missing}")
        self.rows = df.assign(stem=df.index)[["stem", *NEED]].to_dict("records")
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def _read(path, flag, what, stem):
        a = cv2.imread(str(path), flag)
        if a is None:
            raise FileNotFoundError(f"{stem}: cannot read {what}: {path}")
        return a

    def __getitem__(self, i):
        r = self.rows[i]; stem = r["stem"]
        img = cv2.cvtColor(self._read(r["img_path"], cv2.IMREAD_COLOR, "image", stem), cv2.COLOR_BGR2RGB)
        disc = self._read(r["disc_path"], cv2.IMREAD_GRAYSCALE, "disc mask", stem)
        cup = self._read(r["cup_path"], cv2.IMREAD_GRAYSCALE, "cup mask", stem)
        if not (img.shape[:2] == disc.shape == cup.shape):
            raise ValueError(f"{stem}: image {img.shape[:2]} vs masks {disc.shape}/{cup.shape}")
        disc, cup = (disc > 0).astype(np.uint8), (cup > 0).astype(np.uint8)   # {0,255} -> {0,1}
        out = self.transform(image=img, masks=[disc, cup])    # identical geometry for image and both masks
        mask = torch.stack([torch.as_tensor(m) for m in out["masks"]]).float()  # (2,H,W): ch0=OD, ch1=OC
        meta = dict(stem=stem, label=r["label"], source=r["source"], orig_size=int(img.shape[0]))
        return out["image"], mask, meta

def seed_worker(worker_id):
    # Every worker holds a copy of the transform pipeline. Without re-seeding, the copies can draw the
    # same random numbers, and since workers restart each epoch they can repeat them epoch after epoch.
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)
    tf = torch.utils.data.get_worker_info().dataset.transform
    if hasattr(tf, "set_random_seed"):
        tf.set_random_seed(seed)

def make_loaders(frames, train_tf, eval_tf, batch_size=4, num_workers=2, seed=42):
    g = torch.Generator(); g.manual_seed(seed)           # controls the shuffle order
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker)
    train = DataLoader(RimOneDataset(frames["train"], train_tf), shuffle=True, drop_last=True, generator=g, **kw)
    val = DataLoader(RimOneDataset(frames["val"], eval_tf), shuffle=False, **kw)
    test = DataLoader(RimOneDataset(frames["test"], eval_tf), shuffle=False, **kw)
    return train, val, test
