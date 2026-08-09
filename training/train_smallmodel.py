"""
Full Training Pipeline: Smaller Model (12M params)
====================================================
Phase 1: Train on synthetic data with transformed labels
Phase 2: Fine-tune on clinical data with contrast-aware loss

"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.filters import gaussian
from skimage.transform import resize as sk_resize
from sewar import vifp
from model.fullmodel_small import FullModelSmall
from utils.clinical_dataloading import ClinicalLabelledDataset
from utils.balancedLoss import BalancedLoss
from utils.perceptual_loss import PerceptualLoss
from utils.otherUtils import compute_ncc
import wandb


# ══════════════════════════════════════════════════════════════════════
#                        CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

synthetic_images      = "data/train/tumor_images_final.npy"
synthetic_labels      = "data/train/tumor_labels_elastography_image.npy"
clinical_rf_folder    = "/home/deeplearningtower/Documents/TUFFC_2022_Bi_Directional_elasto"
clinical_label_folder = "data/RivazElastographyLabels"
clinical_mask_folder  = "data/RFLigamentMasks"

PHASE1_EPOCHS     = 50
PHASE1_START_LR   = 3e-4
PHASE1_MAX_LR     = 1e-3
PHASE1_BATCH_SIZE = 4
PHASE1_PATIENCE   = 50

PHASE2_EPOCHS     = 80
PHASE2_LR         = 3e-5
PHASE2_BATCH_SIZE = 4

USE_OHEM          = True
OHEM_WARMUP       = 5
OHEM_TOP_K        = 0.7
USE_MIXUP         = True
MIXUP_ALPHA       = 0.2
T_0               = 10     
T_MULT            = 1

WEIGHT_L1         = 1.0
WEIGHT_SSIM       = 0.5
WEIGHT_GDL        = 0.5
WEIGHT_TV         = 0.002
WEIGHT_MEAN       = 0.2
WEIGHT_CONTRAST   = 0.3
WEIGHT_RANGE      = 0.1
WEIGHT_PERCEPTUAL = 0.1  

CROP_TOP    = 40
CROP_BOTTOM = 170
BLUR_SIGMA  = 2
SPECKLE_STR = 0.35
SPECKLE_COR = 0.8

NUM_ECHO_TYPES        = 3
CLINICAL_VAL_COUNT    = 100
CLINICAL_TEST_COUNT   = 100

WANDB_PROJECT  = "elastography"
WANDB_RUN_NAME = "small_model_full_pipeline"


# ══════════════════════════════════════════════════════════════════════
#           SYNTHETIC DATASET WITH LABEL TRANSFORMATION
# ══════════════════════════════════════════════════════════════════════

def synth_to_clinical(label, seed=None):
    """Transform a synthetic (220, 200) label to look clinical."""
    rng = np.random.RandomState(seed)
    low, high = np.percentile(label, 5), np.percentile(label, 95)
    label = np.clip(label, low, high)
    label = (label - low) / (high - low + 1e-8)
    cropped = label[CROP_TOP:CROP_BOTTOM, :]
    resized = sk_resize(cropped, (220, 200), order=1, preserve_range=True).astype(np.float32)
    blurred = gaussian(resized, sigma=BLUR_SIGMA)
    raw_noise = rng.randn(*blurred.shape).astype(np.float32)
    correlated_noise = gaussian(raw_noise, sigma=SPECKLE_COR)
    result = blurred * (1 + SPECKLE_STR * correlated_noise)
    return np.clip(result, 0, 1).astype(np.float32)


class TransformedSyntheticDataset(Dataset):
    def __init__(self, images_path, labels_path, augment=False, transform_labels=True):
        self.images = np.load(images_path, mmap_mode='r')
        self.labels = np.load(labels_path, mmap_mode='r')
        self.augment = augment
        self.transform_labels = transform_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy()
        lbl = self.labels[idx].copy()
        lbl = lbl.astype(np.float32) / 255.0
        lbl = np.clip(lbl, 0, 1)

        if self.transform_labels:
            lbl = synth_to_clinical(lbl, seed=idx)

        if self.augment:
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :].copy()
                lbl = lbl[:, ::-1].copy()
            if np.random.rand() < 0.5:
                noise_std = np.random.uniform(0.001, 0.02) * np.abs(img).max()
                img = img + np.random.randn(*img.shape).astype(np.float32) * noise_std
            if np.random.rand() < 0.5:
                img = img * np.random.uniform(0.85, 1.15)

        img = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
        img = img / (img.abs().max() + 1e-8)
        lbl = torch.from_numpy(lbl.copy()).unsqueeze(0).float().clamp(0.0, 1.0)
        return img, lbl


# ══════════════════════════════════════════════════════════════════════
#                   LOSS CLASSES
# ══════════════════════════════════════════════════════════════════════

class ContrastAwareLoss(nn.Module):
    def __init__(self, weight_contrast=0.3, weight_range=0.1, patch_size=16):
        super().__init__()
        self.weight_contrast = weight_contrast
        self.weight_range = weight_range
        self.patch_size = patch_size

    def local_variance(self, x, patch_size):
        padding = patch_size // 2
        local_mean = F.avg_pool2d(x, patch_size, stride=1, padding=padding)
        local_mean_sq = F.avg_pool2d(x ** 2, patch_size, stride=1, padding=padding)
        min_h = min(local_mean.shape[2], local_mean_sq.shape[2], x.shape[2])
        min_w = min(local_mean.shape[3], local_mean_sq.shape[3], x.shape[3])
        local_mean = local_mean[:, :, :min_h, :min_w]
        local_mean_sq = local_mean_sq[:, :, :min_h, :min_w]
        return (local_mean_sq - local_mean ** 2).clamp(min=0)

    def forward(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        if self.weight_contrast > 0:
            pred_var = self.local_variance(pred, self.patch_size)
            target_var = self.local_variance(target, self.patch_size)
            loss = loss + self.weight_contrast * F.l1_loss(
                torch.sqrt(pred_var + 1e-6), torch.sqrt(target_var + 1e-6))
        if self.weight_range > 0:
            B = pred.shape[0]
            range_loss = torch.tensor(0.0, device=pred.device)
            for i in range(B):
                range_loss = range_loss + (
                    (pred[i].min() - target[i].min()).abs() +
                    (pred[i].max() - target[i].max()).abs()
                )
            loss = loss + self.weight_range * range_loss / B
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, *losses):
        super().__init__()
        self.losses = nn.ModuleList(losses)

    def forward(self, pred, target):
        total = torch.tensor(0.0, device=pred.device)
        for loss_fn in self.losses:
            total = total + loss_fn(pred, target)
        return total


class OHEMWrapper(nn.Module):
    def __init__(self, criterion, top_k=0.7):
        super().__init__()
        self.criterion = criterion
        self.top_k = top_k

    def forward(self, pred, target, use_ohem=True):
        if not use_ohem or pred.size(0) <= 1:
            return self.criterion(pred, target)
        per_sample = [self.criterion(pred[i:i+1], target[i:i+1]) for i in range(pred.size(0))]
        losses = torch.stack(per_sample)
        k = max(1, int(self.top_k * len(losses)))
        topk_losses, _ = torch.topk(losses, k)
        return topk_losses.mean()


# ══════════════════════════════════════════════════════════════════════
#                        UTILITIES
# ══════════════════════════════════════════════════════════════════════

def mixup_data(x, y, alpha=0.2):
    lam = max(np.random.beta(alpha, alpha), 1 - np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], lam * y + (1 - lam) * y[index], lam


def split_by_phantom(dataset, num_echo_types=3, ratios=(0.8, 0.1, 0.1), seed=42):
    total = len(dataset)
    num_phantoms = total // num_echo_types
    assert num_phantoms * num_echo_types == total
    rng = np.random.RandomState(seed)
    phantom_ids = np.arange(num_phantoms)
    rng.shuffle(phantom_ids)
    n_train = int(np.floor(ratios[0] * num_phantoms))
    n_val   = int(np.floor(ratios[1] * num_phantoms))

    def to_idx(phantoms):
        idx = []
        for p in phantoms:
            idx.extend(range(p * num_echo_types, (p + 1) * num_echo_types))
        return idx

    train_p = phantom_ids[:n_train]
    val_p   = phantom_ids[n_train:n_train + n_val]
    test_p  = phantom_ids[n_train + n_val:]
    print(f"Split: {len(train_p)} phantoms ({len(to_idx(train_p))} samples) train, "
          f"{len(val_p)} ({len(to_idx(val_p))}) val, "
          f"{len(test_p)} ({len(to_idx(test_p))}) test")
    return (Subset(dataset, to_idx(train_p)),
            Subset(dataset, to_idx(val_p)),
            Subset(dataset, to_idx(test_p)))


def compute_metrics(pred_np, gt_np):
    pred_np = np.clip(pred_np, 0.0, 1.0)
    gt_np   = np.clip(gt_np,   0.0, 1.0)
    p255 = (pred_np * 255).astype(np.uint8)
    g255 = (gt_np   * 255).astype(np.uint8)
    mse = np.mean((pred_np - gt_np) ** 2)
    mae = np.mean(np.abs(pred_np - gt_np))
    pred_std, gt_std = np.std(pred_np), np.std(gt_np)
    pred_range = pred_np.max() - pred_np.min()
    gt_range = gt_np.max() - gt_np.min()
    return {
        'psnr': peak_signal_noise_ratio(g255, p255, data_range=255),
        'ssim': structural_similarity(g255, p255, data_range=255, channel_axis=None),
        'vif':  vifp(g255, p255),
        'ncc':  compute_ncc(gt_np, pred_np),
        'mse':  float(mse), 'rmse': float(np.sqrt(mse)), 'mae': float(mae),
        'contrast_ratio':  float(pred_std / (gt_std + 1e-8)),
        'brightness_diff': float(np.mean(pred_np) - np.mean(gt_np)),
        'range_ratio':     float(pred_range / (gt_range + 1e-8)),
    }

METRIC_GROUPS = {
    'Image Quality': ['psnr', 'ssim', 'vif', 'ncc'],
    'Pixel Error':   ['mse', 'rmse', 'mae'],
    'Contrast':      ['contrast_ratio', 'brightness_diff', 'range_ratio'],
}

HISTOGRAM_METRICS = {
    'ncc':  {'xlabel': 'NCC',  'good_thresh': 0.8, 'bad_thresh': 0.3},
    'ssim': {'xlabel': 'SSIM', 'good_thresh': 0.7, 'bad_thresh': 0.3},
    'psnr': {'xlabel': 'PSNR (dB)', 'good_thresh': 25, 'bad_thresh': 15},
    'mae':  {'xlabel': 'MAE', 'good_thresh': None, 'bad_thresh': None},
    'contrast_ratio':  {'xlabel': 'Contrast Ratio', 'good_thresh': None, 'bad_thresh': None},
    'brightness_diff': {'xlabel': 'Brightness Diff', 'good_thresh': None, 'bad_thresh': None},
}


def print_metrics(metrics_list, label=""):
    print(f"\n=== {label} Results ({len(metrics_list)} samples) ===")
    for group_name, keys in METRIC_GROUPS.items():
        print(f"\n  {group_name}:")
        for key in keys:
            vals = [m[key] for m in metrics_list]
            print(f"    {key:20s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
                  f"min={np.min(vals):.4f}, max={np.max(vals):.4f}, "
                  f"median={np.median(vals):.4f}")


def evaluate(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if torch.isnan(x).any() or torch.isinf(x).any():
                continue
            total += criterion(model(x), y).item()
            n += 1
    return total / max(n, 1)


def log_metric_histogram(values, metric_name, metric_info, label, epoch, save_dir):
    arr = np.array(values, dtype=np.float64)
    mean_val, std_val, median_val = np.mean(arr), np.std(arr), np.median(arr)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts, bins, patches = ax.hist(arr, bins=30, alpha=0.7, color='#4C72B0', edgecolor='white')
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < mean_val - std_val:
            patch.set_facecolor('#E74C3C')
        elif left_edge > mean_val + std_val:
            patch.set_facecolor('#2ECC71')
    ax.axvline(mean_val, color='black', linewidth=2, label=f'Mean = {mean_val:.4f}')
    ax.axvline(median_val, color='orange', linewidth=2, linestyle='--', label=f'Median = {median_val:.4f}')
    ax.axvspan(mean_val - std_val, mean_val + std_val, alpha=0.15, color='gray', label=f'±1 std ({std_val:.4f})')
    good_t, bad_t = metric_info.get('good_thresh'), metric_info.get('bad_thresh')
    if good_t is not None:
        ax.axvline(good_t, color='green', linewidth=1.5, linestyle=':')
    if bad_t is not None:
        ax.axvline(bad_t, color='red', linewidth=1.5, linestyle=':')
    ax.set_xlabel(metric_info.get('xlabel', metric_name)); ax.set_ylabel('Count')
    ax.set_title(f'{label} — {metric_name.upper()} (Epoch {epoch})\n'
                 f'Mean={mean_val:.4f}, Std={std_val:.4f}, Min={arr.min():.4f}, Max={arr.max():.4f}')
    ax.legend(fontsize=9)
    margin = max(4 * std_val, 0.01)
    ax.set_xlim(mean_val - margin, mean_val + margin)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{metric_name}_histogram_epoch{epoch}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    return path


def log_all_histograms(all_metrics, label, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    wandb_log = {}
    for mname, minfo in HISTOGRAM_METRICS.items():
        vals = [m[mname] for m in all_metrics]
        arr = np.array(vals)
        path = log_metric_histogram(vals, mname, minfo, label, epoch, save_dir)
        wandb_log[f"{label}/{mname}_histogram"] = wandb.Image(path)
        wandb_log[f"{label}/{mname}_mean"] = float(np.mean(arr))
        wandb_log[f"{label}/{mname}_std"] = float(np.std(arr))
        wandb_log[f"{label}/{mname}_median"] = float(np.median(arr))
    # Summary grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, mname in zip(axes.flat, ['ncc', 'ssim', 'psnr', 'mae', 'contrast_ratio', 'brightness_diff']):
        vals = np.array([m[mname] for m in all_metrics])
        ax.hist(vals, bins=25, alpha=0.7, color='#4C72B0', edgecolor='white')
        ax.axvline(np.mean(vals), color='black', linewidth=2)
        ax.axvline(np.median(vals), color='orange', linewidth=1.5, linestyle='--')
        ax.set_title(f'{mname.upper()}\nmean={np.mean(vals):.4f} med={np.median(vals):.4f}', fontsize=10)
    plt.suptitle(f'{label} — Metrics Summary (Epoch {epoch})', fontsize=14)
    plt.tight_layout()
    spath = os.path.join(save_dir, f"summary_epoch{epoch}.png")
    plt.savefig(spath, dpi=150, bbox_inches='tight'); plt.close()
    wandb_log[f"{label}/summary"] = wandb.Image(spath)
    wandb.log(wandb_log)


def run_test(model, loader, device, save_dir, label="Test",
             log_wandb=True, epoch=None, test_files=None):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    model.eval()
    all_metrics, all_preds, all_gts, wandb_images = [], [], [], []

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_np = np.clip(pred.squeeze().cpu().numpy(), 0, 1)
            gt_np = np.clip(y.squeeze().cpu().numpy(), 0, 1)
            m = compute_metrics(pred_np, gt_np)
            all_metrics.append(m); all_preds.append(pred_np); all_gts.append(gt_np)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(pred_np, cmap='viridis'); axes[0].set_title(f'Pred #{idx}'); axes[0].axis('off')
            axes[1].imshow(gt_np, cmap='viridis'); axes[1].set_title(f'GT #{idx}'); axes[1].axis('off')
            axes[2].imshow(np.abs(pred_np - gt_np), cmap='hot', vmin=0, vmax=0.5)
            axes[2].set_title('|Diff|'); axes[2].axis('off')
            plt.suptitle(f'SSIM={m["ssim"]:.3f} PSNR={m["psnr"]:.1f}dB NCC={m["ncc"]:.3f} '
                         f'MAE={m["mae"]:.3f} CR={m["contrast_ratio"]:.2f}', fontsize=9)
            img_path = os.path.join(save_dir, "images", f"result_{idx:03d}.png")
            plt.savefig(img_path, dpi=150, bbox_inches='tight'); plt.close()
            if log_wandb and idx < 50:
                wandb_images.append(wandb.Image(img_path,
                    caption=f"#{idx} NCC={m['ncc']:.3f} SSIM={m['ssim']:.3f} CR={m['contrast_ratio']:.2f}"))

    np.save(os.path.join(save_dir, "all_preds.npy"), np.stack(all_preds))
    np.save(os.path.join(save_dir, "all_gts.npy"), np.stack(all_gts))
    per_sample = {k: np.array([m[k] for m in all_metrics]) for k in all_metrics[0]}
    np.savez(os.path.join(save_dir, "per_sample_metrics.npz"), **per_sample)

    print_metrics(all_metrics, label)
    for rank_metric in ['ncc', 'ssim']:
        arr = np.array([m[rank_metric] for m in all_metrics])
        worst = np.argsort(arr)[:10]
        print(f"\n  Worst 10 by {rank_metric.upper()}:")
        for r, wi in enumerate(worst):
            fname = test_files[wi] if test_files else f"idx {wi}"
            m = all_metrics[wi]
            print(f"    {r+1}. idx={wi} {rank_metric}={arr[wi]:.3f} NCC={m['ncc']:.3f} "
                  f"SSIM={m['ssim']:.3f} CR={m['contrast_ratio']:.2f} file={fname}")

    if log_wandb:
        summary = {}
        for k in all_metrics[0]:
            vals = [m[k] for m in all_metrics]
            summary[f"{label}/{k}_mean"] = float(np.mean(vals))
            summary[f"{label}/{k}_std"] = float(np.std(vals))
            summary[f"{label}/{k}_median"] = float(np.median(vals))
        wandb.log(summary)
        if wandb_images:
            wandb.log({f"{label}_predictions": wandb_images})
        log_all_histograms(all_metrics, label, epoch or 0, os.path.join(save_dir, "histograms"))

    return all_metrics


# ══════════════════════════════════════════════════════════════════════
#                          MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=WANDB_PROJECT, name=WANDB_RUN_NAME,
        config={
            "model": "FullModelSmall (12M params)",
            "phase1_epochs": PHASE1_EPOCHS, "phase2_epochs": PHASE2_EPOCHS,
            "phase1_max_lr": PHASE1_MAX_LR, "phase2_lr": PHASE2_LR,
            "weight_perceptual": WEIGHT_PERCEPTUAL,
            "weight_contrast": WEIGHT_CONTRAST, "weight_range": WEIGHT_RANGE,
            "weight_tv": WEIGHT_TV, "weight_mean": WEIGHT_MEAN,
            "use_ohem": USE_OHEM, "use_mixup": USE_MIXUP,
            "approach": "small_model_full_pipeline",
        }
    )

    # ══════════════════════════════════════════════════════════════
    #                SYNTHETIC DATA (PHASE 1)
    # ══════════════════════════════════════════════════════════════

    syn_train_ds = TransformedSyntheticDataset(synthetic_images, synthetic_labels,
                                                augment=True, transform_labels=True)
    syn_eval_ds  = TransformedSyntheticDataset(synthetic_images, synthetic_labels,
                                                augment=False, transform_labels=True)

    syn_train, _, _      = split_by_phantom(syn_train_ds, NUM_ECHO_TYPES, seed=42)
    _, syn_val, syn_test = split_by_phantom(syn_eval_ds, NUM_ECHO_TYPES, seed=42)

    syn_train_loader = DataLoader(syn_train, batch_size=PHASE1_BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True)
    syn_val_loader   = DataLoader(syn_val, batch_size=PHASE1_BATCH_SIZE, shuffle=False)
    syn_test_loader  = DataLoader(syn_test, batch_size=1, shuffle=False)

    # ══════════════════════════════════════════════════════════════
    #                CLINICAL DATA (PHASE 2)
    # ══════════════════════════════════════════════════════════════

    clinical_full = ClinicalLabelledDataset(
        clinical_rf_folder, clinical_label_folder,
        mask_folder=clinical_mask_folder, augment=False
    )
    num_clinical = len(clinical_full)
    rng = np.random.RandomState(99)
    idx = np.arange(num_clinical)
    rng.shuffle(idx)

    train_idx = idx[:num_clinical - CLINICAL_VAL_COUNT - CLINICAL_TEST_COUNT].tolist()
    val_idx   = idx[num_clinical - CLINICAL_VAL_COUNT - CLINICAL_TEST_COUNT
                    : num_clinical - CLINICAL_TEST_COUNT].tolist()
    test_idx  = idx[num_clinical - CLINICAL_TEST_COUNT:].tolist()

    clinical_train_aug = ClinicalLabelledDataset(
        clinical_rf_folder, clinical_label_folder,
        mask_folder=clinical_mask_folder, augment=True
    )
    clin_train_ds = Subset(clinical_train_aug, train_idx)
    clin_val_ds   = Subset(clinical_full, val_idx)
    clin_test_ds  = Subset(clinical_full, test_idx)

    clin_train_loader = DataLoader(clin_train_ds, batch_size=PHASE2_BATCH_SIZE, shuffle=True,
                                   num_workers=0, pin_memory=True)
    clin_val_loader   = DataLoader(clin_val_ds, batch_size=1, shuffle=False, num_workers=0)
    clin_test_loader  = DataLoader(clin_test_ds, batch_size=1, shuffle=False, num_workers=0)

    test_files = [clinical_full.rf_files[i] for i in test_idx]
    print(f"Clinical: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    # ══════════════════════════════════════════════════════════════
    #                    MODEL
    # ══════════════════════════════════════════════════════════════

    model = FullModelSmall()
    def init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.apply(init_weights)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} total params ({total_params/1e6:.1f}M)")
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    wandb.watch(model, log="gradients", log_freq=200)

    # Phase 1 loss (no contrast or perceptual — those are Phase 2 only)
    p1_criterion = BalancedLoss(
        weight_l1=1.0, weight_ssim=0.5, weight_gdl=0.5,
        weight_tv=0.01, weight_mean=0.1
    )

    # ══════════════════════════════════════════════════════════════
    #       PHASE 1: SYNTHETIC PRE-TRAINING
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 60}")
    print("  PHASE 1: Synthetic training (transformed labels)")
    print(f"  Model: FullModelSmall ({total_params/1e6:.1f}M params)")
    print(f"{'=' * 60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=PHASE1_START_LR)
    scheduler = OneCycleLR(
        optimizer, max_lr=PHASE1_MAX_LR,
        total_steps=len(syn_train_loader) * PHASE1_EPOCHS,
        pct_start=0.1, anneal_strategy='cos', cycle_momentum=False
    )

    best_val = float('inf')
    early_stop_counter = 0
    best_epoch_p1 = 0

    for epoch in range(PHASE1_EPOCHS):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(syn_train_loader, desc=f"P1 Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = p1_criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(syn_train_loader)
        val_loss = evaluate(model, syn_val_loader, p1_criterion, device)

        wandb.log({
            "phase": 1, "epoch": epoch + 1,
            "p1/train_loss": train_loss, "p1/val_loss": val_loss,
            "p1/lr": optimizer.param_groups[0]['lr'],
        })

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                for x, y in syn_val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x); break
                wandb.log({
                    "p1/prediction": wandb.Image(np.clip(pred[0].squeeze().cpu().numpy(), 0, 1)),
                    "p1/target": wandb.Image(np.clip(y[0].squeeze().cpu().numpy(), 0, 1)),
                })

        if val_loss < best_val:
            best_val = val_loss
            best_epoch_p1 = epoch
            early_stop_counter = 0
            torch.save(model.state_dict(), f'model/bestmodel/p1_small_epoch{epoch}.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= PHASE1_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"P1 Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

    # Load best Phase 1
    print(f"\nLoading best Phase 1 model (epoch {best_epoch_p1})")
    model.load_state_dict(torch.load(f'model/bestmodel/p1_small_epoch{best_epoch_p1}.pth',
                                      map_location=device, weights_only=True))

    print("\nPhase 1 synthetic test...")
    run_test(model, syn_test_loader, device, "results/p1_small_synthetic",
             "P1 Synthetic (small)", epoch=best_epoch_p1)

    print("\nPhase 1 on clinical (before fine-tuning)...")
    run_test(model, clin_test_loader, device, "results/p1_small_on_clinical",
             "P1 on Clinical (no FT)", epoch=0, test_files=test_files)

    # ══════════════════════════════════════════════════════════════
    #       PHASE 2: CLINICAL FINE-TUNING
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 60}")
    print("  PHASE 2: Clinical fine-tuning (contrast-aware + perceptual)")
    print(f"{'=' * 60}")

    # Phase 2 loss: base + contrast + perceptual
    base_loss = BalancedLoss(
        weight_l1=WEIGHT_L1, weight_ssim=WEIGHT_SSIM,
        weight_gdl=WEIGHT_GDL, weight_tv=WEIGHT_TV, weight_mean=WEIGHT_MEAN
    )
    contrast_loss = ContrastAwareLoss(
        weight_contrast=WEIGHT_CONTRAST, weight_range=WEIGHT_RANGE
    )
    perceptual_loss = PerceptualLoss(weight=WEIGHT_PERCEPTUAL).to(device)

    combined_loss = CombinedLoss(base_loss, contrast_loss, perceptual_loss)
    ohem_loss = OHEMWrapper(combined_loss, top_k=OHEM_TOP_K)
    val_criterion = combined_loss  # no OHEM for val

    optimizer = torch.optim.Adam(model.parameters(), lr=PHASE2_LR)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_MULT, eta_min=1e-6)

    best_val_p2 = float('inf')
    best_epoch_p2 = 0

    for epoch in range(1, PHASE2_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        num_batches = 0
        use_ohem = USE_OHEM and (epoch > OHEM_WARMUP)

        for x, y in tqdm(clin_train_loader, desc=f"P2 Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            if torch.isnan(x).any() or torch.isinf(x).any():
                continue
            if USE_MIXUP and epoch > 2:
                x, y, lam = mixup_data(x, y, alpha=MIXUP_ALPHA)

            optimizer.zero_grad()
            pred = model(x)
            loss = ohem_loss(pred, y, use_ohem=use_ohem)
            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            train_loss_sum += loss.item()
            num_batches += 1

        scheduler.step(epoch)
        train_loss = train_loss_sum / max(num_batches, 1)
        val_loss = evaluate(model, clin_val_loader, val_criterion, device)

        wandb.log({
            "phase": 2, "epoch": PHASE1_EPOCHS + epoch,
            "p2/train_loss": train_loss, "p2/val_loss": val_loss,
            "p2/lr": optimizer.param_groups[0]['lr'],
        })

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            epoch_metrics = []
            sample_logged = False
            with torch.no_grad():
                for x_t, y_t in clin_val_loader:
                    x_t, y_t = x_t.to(device), y_t.to(device)
                    pred_t = model(x_t)
                    if not sample_logged:
                        wandb.log({
                            "p2/prediction": wandb.Image(np.clip(pred_t[0].squeeze().cpu().numpy(), 0, 1)),
                            "p2/target": wandb.Image(np.clip(y_t[0].squeeze().cpu().numpy(), 0, 1)),
                        })
                        sample_logged = True
                    pred_np = np.clip(pred_t.squeeze().cpu().numpy(), 0, 1)
                    gt_np = np.clip(y_t.squeeze().cpu().numpy(), 0, 1)
                    epoch_metrics.append(compute_metrics(pred_np, gt_np))
            os.makedirs("results/p2_small_histograms", exist_ok=True)
            log_all_histograms(epoch_metrics, "p2/during_training", epoch, "results/p2_small_histograms")

        if val_loss < best_val_p2:
            best_val_p2 = val_loss
            best_epoch_p2 = epoch
            torch.save(model.state_dict(), f'model/bestmodel/p2_small_epoch{epoch}.pth')

        print(f"P2 Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"OHEM={'ON' if use_ohem else 'off'}, LR={optimizer.param_groups[0]['lr']:.2e}")

    # ══════════════════════════════════════════════════════════════
    #              FINAL EVALUATION
    # ══════════════════════════════════════════════════════════════

    print(f"\nLoading best Phase 2 model (epoch {best_epoch_p2})")
    model.load_state_dict(torch.load(f'model/bestmodel/p2_small_epoch{best_epoch_p2}.pth',
                                      map_location=device, weights_only=True))

    print(f"\n{'=' * 60}")
    print("  Final Clinical Evaluation (Small Model)")
    print(f"{'=' * 60}")

    run_test(model, clin_test_loader, device,
             "results/clinical_small_results",
             "Clinical (Small Model)", epoch=best_epoch_p2,
             test_files=test_files)

    print("\nSynthetic performance after fine-tuning...")
    run_test(model, syn_test_loader, device,
             "results/synthetic_small_after_ft",
             "Synthetic (after FT)", epoch=best_epoch_p2)

    with open("results/clinical_small_results/test_files.txt", 'w') as f:
        for fname in test_files:
            f.write(fname + '\n')

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()