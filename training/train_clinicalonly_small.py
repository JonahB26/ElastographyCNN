"""
Clinical-Only Training (No Synthetic Pre-training)
====================================================
Trains FullModelSmall from scratch on clinical data only.
Useful as a baseline to measure how much synthetic pre-training helps.

Uses the same 3-way split, loss functions, and metrics as the
full pipeline for direct comparison.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
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

clinical_rf_folder    = "/home/deeplearningtower/Documents/TUFFC_2022_Bi_Directional_elasto"
clinical_label_folder = "data/RivazElastographyLabels"
clinical_mask_folder  = "data/RFLigamentMasks"

EPOCHS            = 100
LR                = 5e-4
BATCH_SIZE        = 4
T_0               = 20
T_MULT            = 1

WEIGHT_L1         = 1.0
WEIGHT_SSIM       = 0.5
WEIGHT_GDL        = 0.5
WEIGHT_TV         = 0.002
WEIGHT_MEAN       = 0.2
WEIGHT_PERCEPTUAL = 0.2

CLINICAL_TRAIN_CAP = 400
CLINICAL_VAL_COUNT  = 100
CLINICAL_TEST_COUNT = 100

WANDB_PROJECT  = "elastography"
WANDB_RUN_NAME = "clinical_only_small_model"


# ══════════════════════════════════════════════════════════════════════
#                        UTILITIES
# ══════════════════════════════════════════════════════════════════════

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


class CombinedLoss(nn.Module):
    def __init__(self, *losses):
        super().__init__()
        self.losses = nn.ModuleList(losses)

    def forward(self, pred, target):
        total = torch.tensor(0.0, device=pred.device)
        for loss_fn in self.losses:
            total = total + loss_fn(pred, target)
        return total


# ══════════════════════════════════════════════════════════════════════
#                          MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=WANDB_PROJECT, name=WANDB_RUN_NAME,
        config={
            "model": "FullModelSmall (12M)",
            "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
            "weight_perceptual": WEIGHT_PERCEPTUAL,
            "approach": "clinical_only_no_pretraining",
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

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    test_files = [clinical_full.rf_files[i] for i in test_idx]
    print(f"Clinical: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    model = FullModelSmall()
    def init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.apply(init_weights)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params ({total_params/1e6:.1f}M) — training from scratch")

    base_loss = BalancedLoss(
        weight_l1=WEIGHT_L1, weight_ssim=WEIGHT_SSIM,
        weight_gdl=WEIGHT_GDL, weight_tv=WEIGHT_TV, weight_mean=WEIGHT_MEAN
    )
    perceptual_loss = PerceptualLoss(weight=WEIGHT_PERCEPTUAL).to(device)
    criterion = CombinedLoss(base_loss, perceptual_loss)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_MULT, eta_min=1e-6)

    print(f"\n{'=' * 60}")
    print("  Training from scratch on clinical data only")
    print(f"{'=' * 60}")

    best_val = float('inf')
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        num_batches = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            if torch.isnan(x).any() or torch.isinf(x).any():
                continue

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            train_loss_sum += loss.item()
            num_batches += 1

        scheduler.step(epoch)
        train_loss = train_loss_sum / max(num_batches, 1)
        val_loss = evaluate(model, val_loader, criterion, device)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]['lr'],
        })

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                for x_t, y_t in val_loader:
                    x_t, y_t = x_t.to(device), y_t.to(device)
                    pred_t = model(x_t)
                    break
                wandb.log({
                    "prediction": wandb.Image(np.clip(pred_t[0].squeeze().cpu().numpy(), 0, 1)),
                    "target": wandb.Image(np.clip(y_t[0].squeeze().cpu().numpy(), 0, 1)),
                })

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f'model/best_small_clinical/clinical_only_epoch{epoch}.pth')

        print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"LR={optimizer.param_groups[0]['lr']:.2e}")

    print(f"\nLoading best model (epoch {best_epoch})")
    model.load_state_dict(torch.load(f'model/best_small_clinical/clinical_only_epoch{best_epoch}.pth',
                                      map_location=device, weights_only=True))

    print(f"\n{'=' * 60}")
    print("  Final Test Evaluation")
    print(f"{'=' * 60}")

    save_dir = "results/clinical_only_results"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)

    model.eval()
    all_metrics = []
    all_preds = []
    all_gts = []
    wandb_images = []

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_np = np.clip(pred.squeeze().cpu().numpy(), 0, 1)
            gt_np = np.clip(y.squeeze().cpu().numpy(), 0, 1)

            m = compute_metrics(pred_np, gt_np)
            all_metrics.append(m)
            all_preds.append(pred_np)
            all_gts.append(gt_np)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(gt_np, cmap='viridis')
            ax1.set_title('Ground Truth', fontsize=14)
            ax1.axis('off')
            ax2.imshow(pred_np, cmap='viridis')
            ax2.set_title('Prediction', fontsize=14)
            ax2.axis('off')
            plt.tight_layout()
            img_path = os.path.join(save_dir, "images", f"sample_{i:03d}.png")
            plt.savefig(img_path, dpi=200, bbox_inches='tight')
            plt.close()

            wandb_images.append(wandb.Image(
                img_path, caption=f"#{i} NCC={m['ncc']:.3f} SSIM={m['ssim']:.3f}"))

    np.save(os.path.join(save_dir, "all_preds.npy"), np.stack(all_preds))
    np.save(os.path.join(save_dir, "all_gts.npy"), np.stack(all_gts))

    per_sample = {k: np.array([m[k] for m in all_metrics]) for k in all_metrics[0]}
    np.savez(os.path.join(save_dir, "per_sample_metrics.npz"), **per_sample)

    print_metrics(all_metrics, "Clinical-Only Model")

    for rank_metric in ['ncc', 'ssim']:
        arr = np.array([m[rank_metric] for m in all_metrics])
        worst = np.argsort(arr)[:10]
        print(f"\n  Worst 10 by {rank_metric.upper()}:")
        for r, wi in enumerate(worst):
            fname = test_files[wi]
            m = all_metrics[wi]
            print(f"    {r+1}. idx={wi} {rank_metric}={arr[wi]:.3f} NCC={m['ncc']:.3f} "
                  f"SSIM={m['ssim']:.3f} CR={m['contrast_ratio']:.2f} file={fname}")

    wandb.log({"test_predictions": wandb_images})

    ncc_arr = np.array([m['ncc'] for m in all_metrics])
    fig, ax = plt.subplots(figsize=(10, 5))
    counts, bins, patches = ax.hist(ncc_arr, bins=30, alpha=0.7, color='#4C72B0', edgecolor='white')
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < ncc_arr.mean() - ncc_arr.std():
            patch.set_facecolor('#E74C3C')
        elif left_edge > ncc_arr.mean() + ncc_arr.std():
            patch.set_facecolor('#2ECC71')
    ax.axvline(ncc_arr.mean(), color='black', linewidth=2, label=f'Mean = {ncc_arr.mean():.3f}')
    ax.axvline(np.median(ncc_arr), color='orange', linewidth=2, linestyle='--', label=f'Median = {np.median(ncc_arr):.3f}')
    ax.axvspan(ncc_arr.mean() - ncc_arr.std(), ncc_arr.mean() + ncc_arr.std(), alpha=0.15, color='gray', label=f'±1 std ({ncc_arr.std():.3f})')
    ax.axvline(0.8, color='green', linewidth=1.5, linestyle=':', label='Good (0.8)')
    ax.axvline(0.3, color='red', linewidth=1.5, linestyle=':', label='Bad (0.3)')
    ax.set_xlabel('NCC', fontsize=12); ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Clinical-Only Model — NCC Distribution\n'
                 f'Mean={ncc_arr.mean():.3f}, Median={np.median(ncc_arr):.3f}, '
                 f'Std={ncc_arr.std():.3f}, <0.3: {np.sum(ncc_arr < 0.3)}, >0.8: {np.sum(ncc_arr > 0.8)}')
    ax.legend(fontsize=9)
    plt.tight_layout()
    hist_path = os.path.join(save_dir, "ncc_histogram.png")
    plt.savefig(hist_path, dpi=200, bbox_inches='tight'); plt.close()
    wandb.log({"test/ncc_histogram": wandb.Image(hist_path)})

    summary = {}
    for k in all_metrics[0]:
        vals = [m[k] for m in all_metrics]
        summary[f"test/{k}_mean"] = float(np.mean(vals))
        summary[f"test/{k}_median"] = float(np.median(vals))
        summary[f"test/{k}_std"] = float(np.std(vals))
    wandb.log(summary)

    with open(os.path.join(save_dir, "test_files.txt"), 'w') as f:
        for fname in test_files:
            f.write(fname + '\n')

    if os.path.isdir("data/ClinicalData"):
        print(f"\n{'=' * 60}")
        print("  ClinicalData Folder Predictions")
        print(f"{'=' * 60}")
        from utils.predict_clinical_pairs import predict_clinical_pairs
        predict_clinical_pairs(
            model, device,
            clinical_folder="data/ClinicalData",
            mask_folder=clinical_mask_folder,
            save_dir="results/clinical_only_clinical_pairs",
            wandb_label="ClinicalData (clinical-only model)",
            log_wandb=True
        )

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()