"""
Phase 2 Only: Small Model Clinical Fine-tuning
================================================
Loads the best Phase 1 checkpoint from the small model (12M params)
and runs Phase 2 fine-tuning with adjustable perceptual loss weight.

Usage:
    python train_phase2_small_perceptual.py --checkpoint model/bestmodel/p1_small_epoch48.pth --train_cap 400
    python train_phase2_small_perceptual.py --checkpoint model/best_small_fullphase/p1_small_epoch47.pth --train_cap 200
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sewar import vifp
from model.fullmodel_small import FullModelSmall
from utils.clinical_dataloading import ClinicalLabelledDataset
from utils.balancedLoss import BalancedLoss
from utils.perceptual_loss import PerceptualLoss
from utils.otherUtils import compute_ncc
import wandb

# ── Parse command-line args ──
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="model/bestmodel/p1_small_epoch48.pth")
parser.add_argument("--train_cap", type=int, default=400)
parser.add_argument("--perceptual_weight", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=80)
args = parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
#                        CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

clinical_rf_folder    = "/home/deeplearningtower/Documents/TUFFC_2022_Bi_Directional_elasto"
clinical_label_folder = "data/RivazElastographyLabels"
clinical_mask_folder  = "data/RFLigamentMasks"
phase1_checkpoint     = args.checkpoint

PHASE2_EPOCHS     = 120
PHASE2_LR         = 3e-5
PHASE2_BATCH_SIZE = 4

USE_OHEM          = True
OHEM_WARMUP       = 5
OHEM_TOP_K        = 0.7
USE_MIXUP         = True
MIXUP_ALPHA       = 0.2
T_0               = 10
T_MULT            = 1

WEIGHT_PERCEPTUAL = 0.1

WEIGHT_L1         = 1.0
WEIGHT_SSIM       = 0.5
WEIGHT_GDL        = 0.5
WEIGHT_TV         = 0.002
WEIGHT_MEAN       = 0.2
WEIGHT_CONTRAST   = 0.3
WEIGHT_RANGE      = 0.1

CLINICAL_TRAIN_CAP  = args.train_cap
CLINICAL_VAL_COUNT  = 100
CLINICAL_TEST_COUNT = 100

ckpt_short = os.path.basename(os.path.dirname(phase1_checkpoint)) + "_" + os.path.basename(phase1_checkpoint).replace('.pth', '')
WANDB_PROJECT  = "elastography"
WANDB_RUN_NAME = f"p2_{CLINICAL_TRAIN_CAP}pt_{ckpt_short}"


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
            if log_wandb:
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

    cr_arr = np.array([m['contrast_ratio'] for m in all_metrics])
    bd_arr = np.array([m['brightness_diff'] for m in all_metrics])
    print(f"\n  Contrast Summary:")
    print(f"    Contrast Ratio: mean={cr_arr.mean():.3f} ({'under-contrast' if cr_arr.mean() < 0.9 else 'ok'})")
    print(f"    Brightness Diff: mean={bd_arr.mean():.4f} ({'darker' if bd_arr.mean() < -0.02 else 'ok'})")

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
            "model": "FullModelSmall (12M)",
            "phase2_epochs": PHASE2_EPOCHS, "phase2_lr": PHASE2_LR,
            "phase1_checkpoint": phase1_checkpoint,
            "weight_perceptual": WEIGHT_PERCEPTUAL,
            "weight_contrast": WEIGHT_CONTRAST, "weight_range": WEIGHT_RANGE,
            "weight_l1": WEIGHT_L1, "weight_ssim": WEIGHT_SSIM,
            "weight_gdl": WEIGHT_GDL, "weight_tv": WEIGHT_TV,
            "weight_mean": WEIGHT_MEAN,
            "use_ohem": USE_OHEM, "use_mixup": USE_MIXUP,
            "approach": f"small_p2_perceptual_{WEIGHT_PERCEPTUAL}",
        }
    )

    clinical_full = ClinicalLabelledDataset(
        clinical_rf_folder, clinical_label_folder,
        mask_folder=clinical_mask_folder, augment=False
    )
    num_clinical = len(clinical_full)
    rng = np.random.RandomState(99)
    idx = np.arange(num_clinical)
    rng.shuffle(idx)

    train_idx = idx[:num_clinical - CLINICAL_VAL_COUNT - CLINICAL_TEST_COUNT].tolist()
    if CLINICAL_TRAIN_CAP is not None:
        train_idx = train_idx[:CLINICAL_TRAIN_CAP]
    val_idx   = idx[num_clinical - CLINICAL_VAL_COUNT - CLINICAL_TEST_COUNT
                    : num_clinical - CLINICAL_TEST_COUNT].tolist()
    test_idx  = idx[num_clinical - CLINICAL_TEST_COUNT:].tolist()

    clinical_train_aug = ClinicalLabelledDataset(
        clinical_rf_folder, clinical_label_folder,
        mask_folder=clinical_mask_folder, augment=True
    )
    train_ds = Subset(clinical_train_aug, train_idx)
    val_ds   = Subset(clinical_full, val_idx)
    test_ds  = Subset(clinical_full, test_idx)

    train_loader = DataLoader(train_ds, batch_size=PHASE2_BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    test_files = [clinical_full.rf_files[i] for i in test_idx]
    print(f"Clinical: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    model = FullModelSmall().to(device)
    model.load_state_dict(torch.load(phase1_checkpoint, map_location=device, weights_only=True))
    print(f"Loaded checkpoint: {phase1_checkpoint}")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

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
    val_criterion = combined_loss

    print("\n--- Phase 1 model on clinical (before fine-tuning) ---")
    run_test(model, test_loader, device, "results/p2_small_perc_before_ft",
             "P1 on Clinical (no FT)", epoch=0, test_files=test_files)

    if os.path.isdir("ClinicalData"):
        print("\n--- ClinicalData predictions (before fine-tuning) ---")
        from utils.predict_clinical_pairs import predict_clinical_pairs
        predict_clinical_pairs(
            model, device,
            clinical_folder="ClinicalData",
            mask_folder=clinical_mask_folder,
            save_dir="results/p2_small_perc_clinical_pairs_before_ft",
            wandb_label="ClinicalData (before FT)",
            log_wandb=True
        )

    print(f"\n{'=' * 60}")
    print(f"  PHASE 2: Small model, perceptual weight = {WEIGHT_PERCEPTUAL}")
    print(f"{'=' * 60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=PHASE2_LR)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_MULT, eta_min=1e-6)

    best_val = float('inf')
    best_epoch = 0

    for epoch in range(1, PHASE2_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        num_batches = 0
        use_ohem = USE_OHEM and (epoch > OHEM_WARMUP)

        for x, y in tqdm(train_loader, desc=f"P2 Epoch {epoch}"):
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
        val_loss = evaluate(model, val_loader, val_criterion, device)

        wandb.log({
            "epoch": epoch,
            "p2/train_loss": train_loss,
            "p2/val_loss": val_loss,
            "p2/lr": optimizer.param_groups[0]['lr'],
        })

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            epoch_metrics = []
            sample_logged = False
            with torch.no_grad():
                for x_t, y_t in val_loader:
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
            os.makedirs("results/p2_small_perc_histograms", exist_ok=True)
            log_all_histograms(epoch_metrics, "p2/during_training", epoch,
                               "results/p2_small_perc_histograms")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            save_path = f'model/bestmodel/p2_{CLINICAL_TRAIN_CAP}pt_{ckpt_short}_epoch{epoch}.pth'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"OHEM={'ON' if use_ohem else 'off'}, LR={optimizer.param_groups[0]['lr']:.2e}")

    best_path = f'model/bestmodel/p2_{CLINICAL_TRAIN_CAP}pt_{ckpt_short}_epoch{best_epoch}.pth'
    print(f"\nLoading best model (epoch {best_epoch})")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    print(f"\n{'=' * 60}")
    print(f"  Final Clinical Evaluation (perceptual={WEIGHT_PERCEPTUAL})")
    print(f"{'=' * 60}")

    run_test(model, test_loader, device,
             "results/clinical_small_perc_results",
             "Clinical (Small+Perceptual)", epoch=best_epoch,
             test_files=test_files)

    with open("results/clinical_small_perc_results/test_files.txt", 'w') as f:
        for fname in test_files:
            f.write(fname + '\n')

    if os.path.isdir("ClinicalData"):
        print(f"\n{'=' * 60}")
        print("  ClinicalData Folder Predictions (after fine-tuning)")
        print(f"{'=' * 60}")
        from utils.predict_clinical_pairs import predict_clinical_pairs
        predict_clinical_pairs(
            model, device,
            clinical_folder="ClinicalData",
            mask_folder=clinical_mask_folder,
            save_dir="results/p2_small_perc_clinical_pairs_after_ft",
            wandb_label="ClinicalData (after FT)",
            log_wandb=True
        )

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()