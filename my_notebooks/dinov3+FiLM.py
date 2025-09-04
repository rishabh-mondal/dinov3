#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DINOv3 Faster R-CNN + FiLM conditioning with AlphaEarth (AEF) embeddings
- Uses a 64-D AEF vector per region to FiLM-modulate the backbone features
- End-to-end: train, validate, save best, evaluate in-region & OOR

Requirements: torch, torchvision, torchmetrics, numpy, pillow, tqdm
Optional for GeoTIFF: rasterio (preferred) or tifffile
"""

import os
import csv
import math
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchmetrics.detection import MeanAveragePrecision

# =========================
# Configuration
# =========================
# ---- DINOv3 ----
DINOV3_GITHUB_LOCATION = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/dinov3"
DINOV3_LOCATION = os.getenv("DINOV3_LOCATION") or DINOV3_GITHUB_LOCATION
DINO_MODEL_NAME = "dinov3_vitl16"
DINO_WEIGHTS = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

# ---- Data roots ----
UP_ROOT  = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/uttar_pradesh"
BD_ROOT  = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/bangladesh"
PKP_ROOT = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/data/iclr_2026_processed_data/final_data/pak_punjab"

# ---- AlphaEarth rasters (64 bands) ----
AEF_PATHS = {
    "uttar_pradesh": "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/alphaearth/AEF_2024_poly_utm43n-7brick-up.tif",
    "bangladesh":    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/alphaearth/AEF_2024_poly_utm43n-17brick-bangladesh.tif",
    "pak_punjab":    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/Foundation-Models/alphaearth/AEF_2024_poly_utm43n-7brick-pak_punjab.tif",
}

# ---- Training & eval ----
IMAGE_SIZE    = 800
BATCH_SIZE    = 8
NUM_WORKERS   = 8
NUM_EPOCHS    = 10
BACKBONE_LR   = 1e-5
HEAD_LR       = 1e-4
WEIGHT_DECAY  = 0.04
NUM_CLASSES   = 4  # background + 3 kiln classes

# Turn training on/off quickly
DO_TRAIN      = True

BEST_CKPT     = "best_film_val_map50.pth"
RESULTS_CSV   = "region_eval_film.csv"
LOG_DIR       = "runs/brickkiln_dinov3_film"

# =========================
# AlphaEarth loader
# =========================
def load_aef_vector(aef_tif_path: str) -> torch.Tensor:
    """
    Load a 64-D AlphaEarth raster and reduce to a single [64] vector by spatial mean.
    L2-normalize for stability. Tries rasterio, falls back to tifffile.
    """
    arr = None
    try:
        import rasterio
        with rasterio.open(aef_tif_path) as ds:
            arr = ds.read().astype(np.float32)  # [C,H,W]
    except Exception:
        try:
            import tifffile as tiff
            arr = tiff.imread(aef_tif_path).astype(np.float32)  # [H,W,C] or [C,H,W]
            if arr.ndim == 3 and arr.shape[0] != 64 and arr.shape[-1] == 64:
                arr = np.moveaxis(arr, -1, 0)
        except Exception as e:
            raise RuntimeError(f"Failed to read AEF raster: {aef_tif_path}\n{e}")

    assert arr is not None and arr.ndim == 3, f"AEF raster malformed: {aef_tif_path}"
    C = arr.shape[0]
    if C != 64:
        print(f"[WARN] Expected 64 AEF channels; got {C}. Proceeding anyway.")

    v = arr.reshape(C, -1).mean(-1)  # [C]
    v = torch.from_numpy(v)
    v = v / (v.norm(p=2) + 1e-6)     # L2-normalize
    return v.float()                 # [64]

# =========================
# Dataset
# =========================
class BrickKilnDataset(Dataset):
    """
    <root>/<split>/{images,labels}
    YOLO-OBB line: <cls> x1 y1 x2 y2 x3 y3 x4 y4 in [0,1]
    Converted to axis-aligned XYXY for Faster R-CNN.
    Returns: (image_tensor, target_dict, cond_vec[64])
    """
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

    def __init__(self, root: str, split: str, input_size: int = 224,
                 region_name: Optional[str] = None, aef_path: Optional[str] = None):
        self.root = Path(root)
        self.split = split
        cand = self.root if (self.root / "images").is_dir() else (self.root / split)
        self.img_dir = cand / "images"
        self.label_dir = cand / "labels"
        assert self.img_dir.is_dir(), f"Missing images directory: {self.img_dir}"
        assert self.label_dir.is_dir(), f"Missing label directory: {self.label_dir}"

        self.input_size = int(input_size)
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size),
                              interpolation=transforms.InterpolationMode.BILINEAR,
                              antialias=True),
            transforms.ToTensor(),
        ])

        all_files = sorted([f for f in os.listdir(self.img_dir) if Path(f).suffix.lower() in self.IMG_EXTS])
        self.img_files: List[str] = []
        for img_name in tqdm(all_files, desc=f"Verify {split} data"):
            if self._has_valid_annotations(img_name):
                self.img_files.append(img_name)

        # load region-level conditioning vector (simple & fast)
        self.cond_vec = None
        self.region_name = region_name
        if aef_path is not None and Path(aef_path).exists():
            try:
                self.cond_vec = load_aef_vector(aef_path)  # [64]
                print(f"[AEF] Loaded {region_name} vector from {aef_path} | norm={self.cond_vec.norm():.3f}")
            except Exception as e:
                print(f"[WARN] Could not load AEF vector for {region_name}: {e}")

        logging.info(f"[{split}] valid images: {len(self.img_files)} in {self.img_dir}")

    def _has_valid_annotations(self, img_name: str) -> bool:
        label_path = self.label_dir / f"{Path(img_name).stem}.txt"
        if not label_path.exists():
            return False
        with open(label_path, 'r') as f:
            for line in f:
                if len(line.strip().split()) == 9:
                    return True
        return False

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx: int):
        img_name = self.img_files[idx]
        img_path = self.img_dir / img_name
        label_path = self.label_dir / f"{Path(img_name).stem}.txt"

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        _, Ht, Wt = img_tensor.shape

        boxes, labels = [], []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 9:
                    continue
                cls_id = int(float(parts[0])) + 1  # shift to 1..3 (0 reserved for background)
                obb = np.array([float(p) for p in parts[1:]], dtype=np.float32)
                xs = obb[0::2] * Wt
                ys = obb[1::2] * Ht
                xmin, ymin = float(np.min(xs)), float(np.min(ys))
                xmax, ymax = float(np.max(xs)), float(np.max(ys))
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(cls_id)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        cond = self.cond_vec if self.cond_vec is not None else torch.zeros(64, dtype=torch.float32)
        return img_tensor, target, cond


def collate_fn(batch):
    batch = [item for item in batch if item[1]["boxes"].shape[0] > 0]
    if not batch:
        return None
    images, targets, conds = list(zip(*batch))
    conds = torch.stack(conds, dim=0)  # [B,64]
    return list(images), list(targets), conds

# =========================
# FiLM Adapter + DINOv3 wrapper
# =========================
class FiLMAdapter(nn.Module):
    def __init__(self, feat_dim: int, cond_dim: int = 64, hidden: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * feat_dim),  # gamma, beta
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        # x: [B,C,H,W], cond: [B,cond_dim]
        gb = self.mlp(cond)                 # [B, 2C]
        gamma, beta = gb.chunk(2, dim=1)    # [B,C], [B,C]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
        beta  = beta.unsqueeze(-1).unsqueeze(-1)   # [B,C,1,1]
        return gamma * x + beta

class DinoV3BackboneWrapper(nn.Module):
    """Return {'0': Tensor[B, C, H/16, W/16]} with out_channels=C, FiLM-modulated by 64-D cond."""
    def __init__(self, dino_model: nn.Module, patch_stride: int = 16, cond_dim: int = 64):
        super().__init__()
        self.dino = dino_model
        self.patch_stride = patch_stride
        self._conditioning = None  # [B,cond_dim] per batch

        C = getattr(dino_model, "embed_dim", None) or getattr(dino_model, "num_features", None)
        if C is None:
            with torch.no_grad():
                x = torch.zeros(1, 3, 32, 32)
                tokens, Ht, Wt = self._get_patch_tokens(x)
                C = tokens.shape[-1]
        self.out_channels = C
        self.film = FiLMAdapter(feat_dim=C, cond_dim=cond_dim, hidden=min(4*C, 1024))

    def set_conditioning(self, conds: torch.Tensor):
        """Store per-sample AlphaEarth conditioning for next forward. conds: [B,64]"""
        self._conditioning = conds

    @torch.no_grad()
    def _maybe_h_w(self, x):
        _, _, H, W = x.shape
        return math.ceil(H / self.patch_stride), math.ceil(W / self.patch_stride)

    def _get_patch_tokens(self, x):
        try:
            out = self.dino.forward_features(x)
            if isinstance(out, dict):
                if "x_norm_patchtokens" in out:
                    tokens = out["x_norm_patchtokens"]
                    Ht = out.get("H") or self._maybe_h_w(x)[0]
                    Wt = out.get("W") or self._maybe_h_w(x)[1]
                    return tokens, Ht, Wt
                if "tokens" in out and out["tokens"] is not None:
                    t = out["tokens"]
                    Ht, Wt = self._maybe_h_w(x)
                    if t.shape[1] == (Ht * Wt + 1):
                        t = t[:, 1:, :]
                    return t, Ht, Wt
            if isinstance(out, torch.Tensor):
                t = out
                Ht, Wt = self._maybe_h_w(x)
                N = Ht * Wt
                if t.shape[1] == N + 1:
                    t = t[:, 1:, :]
                elif t.shape[1] != N:
                    N = t.shape[1]
                    Wt = int(round(math.sqrt(N)))
                    Ht = N // Wt
                return t, Ht, Wt
        except Exception:
            pass

        if hasattr(self.dino, "get_intermediate_layers"):
            t = self.dino.get_intermediate_layers(x, n=1, return_class_token=False)[0]
            Ht, Wt = self._maybe_h_w(x)
            return t, Ht, Wt

        t = self.dino(x)
        Ht, Wt = self._maybe_h_w(x)
        if t.dim() == 3 and t.shape[1] == (Ht * Wt + 1):
            t = t[:, 1:, :]
        return t, Ht, Wt

    def forward(self, x: torch.Tensor):
        tokens, Ht, Wt = self._get_patch_tokens(x)     # [B, N, C]
        B, N, C = tokens.shape
        feat = tokens.transpose(1, 2).contiguous().view(B, C, Ht, Wt)  # [B,C,H,W]

        # FiLM modulation
        if self._conditioning is None:
            cond = torch.zeros(B, 64, device=feat.device, dtype=feat.dtype)
        else:
            cond = self._conditioning.to(feat.device, dtype=feat.dtype)
            if cond.dim() != 2 or cond.shape[0] != B:
                raise RuntimeError(f"Conditioning shape mismatch. Got {cond.shape}, need [B,64] for B={B}.")
        feat = self.film(feat, cond)
        return {"0": feat}

def create_model(dino_model: nn.Module, num_classes: int, image_size: int = 800) -> FasterRCNN:
    backbone = DinoV3BackboneWrapper(dino_model, patch_stride=16, cond_dim=64)
    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        min_size=image_size,
        max_size=image_size,
    )
    return model

# =========================
# Train / Validate
# =========================
def get_group_lrs(optimizer):
    return [pg.get("lr", 0.0) for pg in optimizer.param_groups]

def train_one_epoch(model, optimizer, data_loader, device, writer, epoch, global_step):
    model.train()
    total_loss = 0.0
    steps = 0
    for batch in tqdm(data_loader, desc=f"Training epoch {epoch+1}"):
        if batch is None:
            continue
        images, targets, conds = batch
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # set per-batch conditioning
        model.backbone.set_conditioning(conds.to(device))

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        optimizer.step()

        # logging
        if writer is not None:
            writer.add_scalar("train/loss_total_step", float(losses.item()), global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f"train/{k}_step", float(v.item()), global_step)

        total_loss += losses.item()
        steps += 1
        global_step += 1

    # per-epoch LR logging
    if writer is not None:
        for i, lr in enumerate(get_group_lrs(optimizer)):
            writer.add_scalar(f"lr/group_{i}", float(lr), epoch)

        avg_loss = total_loss / max(1, steps)
        writer.add_scalar("train/loss_total_epoch", float(avg_loss), epoch)
    else:
        avg_loss = total_loss / max(1, steps)
    return avg_loss, global_step

@torch.no_grad()
def validate(model, data_loader, device, writer, epoch):
    model.eval()
    metric_mc = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=False)
    metric_c  = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)

    for batch in tqdm(data_loader, desc=f"Validation epoch {epoch+1}"):
        if batch is None:
            continue
        images, targets, conds = batch
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        model.backbone.set_conditioning(conds.to(device))

        outputs = model(images)
        outputs = [{k: v.detach().cpu() for k, v in o.items()} for o in outputs]
        targets = [{k: v.detach().cpu() for k, v in t.items()} for t in targets]
        metric_mc.update(outputs, targets)
        metric_c.update(outputs, targets)

    res_mc = metric_mc.compute()
    res_c  = metric_c.compute()

    map_all = float(res_mc.get("map", torch.tensor(0.0)))
    map_50  = float(res_mc.get("map_50", torch.tensor(0.0)))
    print(f"Validation Results - mAP: {map_all:.4f}, mAP@50: {map_50:.4f}")

    if writer is not None:
        writer.add_scalar("val/mAP_all",  map_all, epoch)
        writer.add_scalar("val/mAP_50",   map_50,  epoch)

        # per-class mAP@50 (optional)
        if "classes" in res_c and "map_per_class" in res_c:
            cls_ids = res_c["classes"].tolist()
            mpc = res_c["map_per_class"].tolist()
            for cid, val in zip(cls_ids, mpc):
                writer.add_scalar(f"val/mAP50_class_{int(cid)}", float(val), epoch)

    return map_all, map_50

# =========================
# Region Evaluation (IN / OOR)
# =========================
@torch.no_grad()
def evaluate_region(model, root: str, split: str, device,
                    batch_size=16, num_workers=8, image_size=224, title="",
                    results_csv=None, region_name: Optional[str] = None, aef_path: Optional[str] = None):
    ds = BrickKilnDataset(root=root, split=split, input_size=image_size,
                          region_name=region_name, aef_path=aef_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    model.eval()
    metric_class = MeanAveragePrecision(box_format='xyxy', class_metrics=True,  iou_thresholds=[0.5])
    metric_agn   = MeanAveragePrecision(box_format='xyxy', class_metrics=False, iou_thresholds=[0.5])
    metric_multi = MeanAveragePrecision(box_format='xyxy', class_metrics=True,  iou_thresholds=[0.5])

    for batch in tqdm(dl, desc=f"Testing [{title}]"):
        if batch is None:
            continue
        images, targets, conds = batch
        images = [i.to(device) for i in images]
        model.backbone.set_conditioning(conds.to(device))
        preds  = model(images)

        preds_cpu = [{k: v.to('cpu') for k, v in p.items()} for p in preds]
        tgts_cpu  = [{k: v.to('cpu') for k, v in t.items()} for t in targets]

        metric_class.update(preds_cpu, tgts_cpu)
        preds_agn = [{'boxes': p['boxes'], 'scores': p['scores'], 'labels': torch.ones_like(p['labels'])} for p in preds_cpu]
        tgts_agn  = [{'boxes': t['boxes'], 'labels': torch.ones_like(t['labels'])} for t in tgts_cpu]
        metric_agn.update(preds_agn, tgts_agn)
        metric_multi.update(preds_cpu, tgts_cpu)

    res_class = metric_class.compute()
    res_agn   = metric_agn.compute()
    res_multi = metric_multi.compute()

    ca_map50 = float(res_agn['map_50']) * 100.0
    mc_map50 = float(res_multi['map']) * 100.0

    classes = res_class.get('classes', torch.tensor([])).tolist() if 'classes' in res_class else []
    mpc     = res_class.get('map_per_class', torch.tensor([])).tolist() if 'map_per_class' in res_class else []
    per_cls = {int(c): v * 100.0 for c, v in zip(classes, mpc)}
    def g(k): return per_cls.get(k, 0.0)

    print("\n" + "=" * 84)
    print(f" Region: {title}")
    print("=" * 84)
    print(f"{'CA mAP@50':<12}{'MC mAP@50':<12}{'CFCBK@50':<12}{'FCBK@50':<12}{'Zigzag@50':<12}")
    print("-" * 84)
    print(f"{ca_map50:<12.2f}{mc_map50:<12.2f}{g(1):<12.2f}{g(2):<12.2f}{g(3):<12.2f}")
    print("=" * 84 + "\n")

    if results_csv is not None:
        is_new = not os.path.exists(results_csv)
        with open(results_csv, "a", newline="") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["Region", "CA_mAP50", "MC_mAP50", "CFCBK_mAP50", "FCBK_mAP50", "Zigzag_mAP50"])
            w.writerow([title, f"{ca_map50:.2f}", f"{mc_map50:.2f}", f"{g(1):.2f}", f"{g(2):.2f}", f"{g(3):.2f}"])

# =========================
# Main
# =========================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print(f"[INFO] DINOv3 location: {DINOV3_LOCATION}")
    dino_model = torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=DINO_MODEL_NAME,
        source="local",
        weights=DINO_WEIGHTS,
        skip_validation=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(dino_model, num_classes=NUM_CLASSES, image_size=IMAGE_SIZE).to(device)

    writer = SummaryWriter(LOG_DIR) if DO_TRAIN else None

    if DO_TRAIN:
        # ====== TRAIN/VAL on PKP (example) ======
        train_ds = BrickKilnDataset(
            root=UP_ROOT, split="train", input_size=IMAGE_SIZE,
            region_name="uttar_pradesh", aef_path=AEF_PATHS.get("uttar_pradesh")
        )
        val_ds   = BrickKilnDataset(
            root=UP_ROOT, split="val",   input_size=IMAGE_SIZE,
            region_name="uttar_pradesh", aef_path=AEF_PATHS.get("uttar_pradesh")
        )

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

        # group params: (a) backbone (dino) small LR, (b) head+film larger LR
        backbone_params, head_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("backbone.dino"):
                backbone_params.append(p)
            else:
                head_params.append(p)

        optimizer = torch.optim.AdamW(
            [{"params": backbone_params, "lr": BACKBONE_LR},
             {"params": head_params,     "lr": HEAD_LR}],
            weight_decay=WEIGHT_DECAY,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        if writer is not None:
            writer.add_text("hparams", f"IMAGE_SIZE={IMAGE_SIZE}, BATCH_SIZE={BATCH_SIZE}, "
                                       f"BACKBONE_LR={BACKBONE_LR}, HEAD_LR={HEAD_LR}, "
                                       f"WEIGHT_DECAY={WEIGHT_DECAY}, EPOCHS={NUM_EPOCHS}")

        best_map50 = -1.0
        global_step = 0
        for epoch in range(NUM_EPOCHS):
            avg_loss, global_step = train_one_epoch(model, optimizer, train_loader, device, writer, epoch, global_step)
            val_map, val_map50 = validate(model, val_loader, device, writer, epoch)

            if writer is not None:
                writer.add_scalar("epoch/train_loss", avg_loss, epoch)
                writer.add_scalar("epoch/val_mAP",    val_map,  epoch)
                writer.add_scalar("epoch/val_mAP50",  val_map50, epoch)

            if val_map50 > best_map50:
                best_map50 = val_map50
                torch.save(model.state_dict(), BEST_CKPT)
                print(f"[CKPT] Saved {BEST_CKPT} at epoch {epoch+1} (val mAP50={best_map50:.4f})")

            lr_scheduler.step()

        if writer is not None:
            writer.flush()
            writer.close()

    # ====== EVAL (load best if exists) ======
    if os.path.exists(BEST_CKPT):
        model.load_state_dict(torch.load(BEST_CKPT, map_location="cpu"))
        print(f"[INFO] Loaded best checkpoint: {BEST_CKPT}")
    model.to(device).eval()

    # IN-REGION: PKP
    evaluate_region(
        model,
        root=UP_ROOT, split="test", device=device,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, image_size=IMAGE_SIZE,
        title="uttar_pradesh — IN-REGION (test)",
        results_csv=RESULTS_CSV,
        region_name="uttar_pradesh", aef_path=AEF_PATHS.get("uttar_pradesh")
    )

    # OOR: UP
    evaluate_region(
        model,
        root=BD_ROOT, split="test", device=device,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, image_size=IMAGE_SIZE,
        title="Bangladesh — OOR (test)",
        results_csv=RESULTS_CSV,
        region_name="Bangladesh", aef_path=AEF_PATHS.get("bangladesh")
    )

    # OOR: Bangladesh
    evaluate_region(
        model,
        root=PKP_ROOT, split="test", device=device,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, image_size=IMAGE_SIZE,
        title="Pak Punjab — OOR (test)",
        results_csv=RESULTS_CSV,
        region_name="pak_punjab", aef_path=AEF_PATHS.get("pak_punjab")
    )


if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=3 nohup python -u dinov3+FiLM.py > film_train_eval_up_global.log 2>&1 &    