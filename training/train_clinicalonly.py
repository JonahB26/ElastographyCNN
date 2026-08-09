"""
Clinical-only training pipeline with Weights & Biases logging.
Trains the model entirely on labelled clinical data (no synthetic pre-training).

Split: 250 test, remaining split 85/15 into train/val.
Logs losses, metrics, and sample images to wandb.
"""

import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sewar import vifp
from model.fullmodel import FullModel
from utils.clinical_dataloading import ClinicalLabelledDataset
from utils.balancedLoss import BalancedLoss
from utils.otherUtils import compute_ncc
import wandb


# ══════════════════════════════════════════════════════════════════════
#                        CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

clinical_rf_folder    = "/home/deeplearningtower/Documents/TUFFC_2022_Bi_Directional_elasto"
clinical_label_folder = "data/ClinicalElastographyLabels"
clinical_mask_folder  = "data/RFLigamentMasks"   # set to None to disable masks
save_dir              = "results/results_clinicalonly"

NUM_EPOCHS    = 50
BATCH_SIZE    = 4
START_LR      = 3e-4
MAX_LR        = 1e-3
PATIENCE      = 30
TEST_COUNT    = 250
VAL_FRACTION  = 0.15
SPLIT_SEED    = 99

# ── wandb config ──
WANDB_PROJECT = "elastography"
WANDB_RUN_NAME = "clinical_only"


# ══════════════════════════════════════════════════════════════════════
#                        METRICS
# ══════════════════════════════════════════════════════════════════════

def dice_score(pred, gt, threshold=0.5):
    pred_bin = (pred >= threshold).astype(np.float32)
    gt_bin   = (gt >= threshold).astype(np.float32)
    intersection = (pred_bin * gt_bin).sum()
    total = pred_bin.sum() + gt_bin.sum()
    if total == 0:
        return 1.0
    return (2.0 * intersection) / total

def mae(pred, gt):
    return np.abs(pred - gt).mean()

def rmse(pred, gt):
    return np.sqrt(((pred - gt) ** 2).mean())

def relative_error(pred, gt):
    denom = np.abs(gt).mean() + 1e-8
    return np.abs(pred - gt).mean() / denom

def contrast_ratio(image, threshold=0.5):
    high = image[image >= threshold]
    low  = image[image < threshold]
    if len(high) == 0 or len(low) == 0:
        return float('nan')
    return high.mean() / (low.mean() + 1e-8)

def compute_all_metrics(pred_np, gt_np):
    pred_np = np.clip(pred_np, 0.0, 1.0)
    gt_np   = np.clip(gt_np,   0.0, 1.0)
    p255 = (pred_np * 255).astype(np.uint8)
    g255 = (gt_np   * 255).astype(np.uint8)
    return {
        'psnr':           peak_signal_noise_ratio(g255, p255, data_range=255),
        'ssim':           structural_similarity(g255, p255, data_range=255, channel_axis=None),
        'vif':            vifp(g255, p255),
        'ncc':            compute_ncc(gt_np, pred_np),
        'dice_0.3':       dice_score(pred_np, gt_np, threshold=0.3),
        'dice_0.5':       dice_score(pred_np, gt_np, threshold=0.5),
        'dice_0.7':       dice_score(pred_np, gt_np, threshold=0.7),
        'mae':            mae(pred_np, gt_np),
        'rmse':           rmse(pred_np, gt_np),
        'relative_error': relative_error(pred_np, gt_np),
        'pred_mean':      pred_np.mean(),
        'gt_mean':        gt_np.mean(),
        'pred_std':       pred_np.std(),
        'gt_std':         gt_np.std(),
        'contrast_pred':  contrast_ratio(pred_np),
        'contrast_gt':    contrast_ratio(gt_np),
    }

def print_summary(metrics_list, label=""):
    print(f"\n{'='*60}")
    print(f"  {label} — {len(metrics_list)} samples")
    print(f"{'='*60}")
    keys = ['psnr', 'ssim', 'vif', 'ncc', 'dice_0.3', 'dice_0.5', 'dice_0.7',
            'mae', 'rmse', 'relative_error']
    for k in keys:
        vals = [m[k] for m in metrics_list if not np.isnan(m[k])]
        if vals:
            print(f"  {k:>18s}:  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                  f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")

def save_metrics_csv(metrics_list, filepath):
    if not metrics_list:
        return
    keys = list(metrics_list[0].keys())
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_idx'] + keys)
        writer.writeheader()
        for i, m in enumerate(metrics_list):
            row = {'sample_idx': i}
            row.update(m)
            writer.writerow(row)
    print(f"  Saved per-sample metrics to {filepath}")


# ══════════════════════════════════════════════════════════════════════
#                        EVALUATION
# ══════════════════════════════════════════════════════════════════════

def run_test(model, loader, device, output_dir, label="Test", file_list=None, log_wandb=True):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    model.eval()
    preds, gts, metrics = [], [], []
    wandb_images = []

    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(loader, desc=f"Evaluating {label}")):
            x, y = x.to(device), y.to(device)
            pred = model(x)

            pred_np = pred.squeeze().cpu().numpy()
            gt_np   = y.squeeze().cpu().numpy()

            m = compute_all_metrics(pred_np, gt_np)
            metrics.append(m)
            preds.append(np.clip(pred_np, 0, 1))
            gts.append(np.clip(gt_np, 0, 1))

            # Save comparison image
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            im0 = axes[0].imshow(preds[-1], cmap='viridis', vmin=0, vmax=1)
            axes[0].set_title('Prediction'); axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(gts[-1], cmap='viridis', vmin=0, vmax=1)
            axes[1].set_title('Ground Truth'); axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            diff = np.abs(preds[-1] - gts[-1])
            im2 = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
            axes[2].set_title('Absolute Difference'); axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            fname = file_list[idx] if file_list else f"sample_{idx}"
            plt.suptitle(
                f'{fname}\nSSIM={m["ssim"]:.3f}  PSNR={m["psnr"]:.1f}dB  '
                f'NCC={m["ncc"]:.3f}  Dice@0.5={m["dice_0.5"]:.3f}  MAE={m["mae"]:.4f}',
                fontsize=10)
            img_path = os.path.join(output_dir, "images", f"result_{idx:03d}.png")
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()

            # Log a subset of images to wandb
            if log_wandb and idx < 50:
                wandb_images.append(wandb.Image(img_path, caption=f"{fname} SSIM={m['ssim']:.3f}"))

    np.save(os.path.join(output_dir, "all_preds.npy"), np.stack(preds))
    np.save(os.path.join(output_dir, "all_gts.npy"), np.stack(gts))
    save_metrics_csv(metrics, os.path.join(output_dir, "per_sample_metrics.csv"))
    print_summary(metrics, label)

    # Save summary text
    keys = ['psnr', 'ssim', 'vif', 'ncc', 'dice_0.3', 'dice_0.5', 'dice_0.7',
            'mae', 'rmse', 'relative_error']
    with open(os.path.join(output_dir, "summary.txt"), 'w') as f:
        f.write(f"{label} — {len(metrics)} samples\n{'='*60}\n")
        for k in keys:
            vals = [m[k] for m in metrics if not np.isnan(m[k])]
            if vals:
                f.write(f"{k:>18s}:  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                        f"min={np.min(vals):.4f}  max={np.max(vals):.4f}\n")

    # Log to wandb
    if log_wandb:
        summary = {}
        for k in keys:
            vals = [m[k] for m in metrics if not np.isnan(m[k])]
            if vals:
                summary[f"test/{k}_mean"] = np.mean(vals)
                summary[f"test/{k}_std"] = np.std(vals)
        wandb.log(summary)
        if wandb_images:
            wandb.log({f"{label}_predictions": wandb_images})

    return metrics


# ══════════════════════════════════════════════════════════════════════
#                          SETUP
# ══════════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(save_dir, exist_ok=True)
os.makedirs("model/bestmodel", exist_ok=True)

# ── Initialize wandb ──
wandb.init(
    project=WANDB_PROJECT,
    name=WANDB_RUN_NAME,
    config={
        "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
        "start_lr": START_LR, "max_lr": MAX_LR,
        "patience": PATIENCE, "test_count": TEST_COUNT,
        "val_fraction": VAL_FRACTION,
        "mask_folder": clinical_mask_folder,
        "architecture": "FullModel (RF1DEncoder + UNet)",
    }
)

# ── Load clinical data ──
full_ds_clean = ClinicalLabelledDataset(
    clinical_rf_folder, clinical_label_folder,
    mask_folder=clinical_mask_folder, augment=False
)
full_ds_aug = ClinicalLabelledDataset(
    clinical_rf_folder, clinical_label_folder,
    mask_folder=clinical_mask_folder, augment=True
)

num_total = len(full_ds_clean)
num_test  = TEST_COUNT
num_remaining = num_total - num_test
num_val   = int(num_remaining * VAL_FRACTION)
num_train = num_remaining - num_val

rng = np.random.RandomState(SPLIT_SEED)
all_idx = np.arange(num_total)
rng.shuffle(all_idx)

test_idx  = all_idx[:num_test].tolist()
val_idx   = all_idx[num_test:num_test + num_val].tolist()
train_idx = all_idx[num_test + num_val:].tolist()

print(f"Clinical split: {num_train} train, {num_val} val, {num_test} test")

train_ds = Subset(full_ds_aug,   train_idx)
val_ds   = Subset(full_ds_clean, val_idx)
test_ds  = Subset(full_ds_clean, test_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False)

# ── Model ──
model = FullModel()
def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
model.apply(init_weights)

torch.cuda.empty_cache()
model = model.to(device)
print(f"GPU memory after model load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

wandb.watch(model, log="gradients", log_freq=100)

criterion = BalancedLoss(
    weight_l1=1.0, weight_ssim=0.5, weight_gdl=0.5,
    weight_tv=0.01, weight_mean=0.1
)

optimizer = torch.optim.Adam(model.parameters(), lr=START_LR)
scheduler = OneCycleLR(
    optimizer, max_lr=MAX_LR,
    total_steps=len(train_loader) * NUM_EPOCHS,
    pct_start=0.1, anneal_strategy='cos', cycle_momentum=False
)


# ══════════════════════════════════════════════════════════════════════
#                        TRAINING
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  Training on clinical data only")
print("=" * 60)

best_val = float('inf')
early_stop_counter = 0
best_epoch = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    num_batches = 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        x, y = x.to(device), y.to(device)
        if torch.isnan(x).any() or torch.isinf(x).any():
            continue

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)

        if torch.isnan(loss):
            print(f"[WARN] NaN loss at epoch {epoch+1}, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        num_batches += 1

    train_loss /= max(num_batches, 1)

    # Validation
    model.eval()
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            if torch.isnan(x).any() or torch.isinf(x).any():
                continue
            val_loss += criterion(model(x), y).item()
            val_batches += 1

    val_loss /= max(val_batches, 1)

    # Log to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "lr": optimizer.param_groups[0]['lr'],
    })

    # Log sample predictions every 5 epochs
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                break
            pred_np = pred[0].squeeze().cpu().numpy()
            gt_np = y[0].squeeze().cpu().numpy()
            wandb.log({
                "val_prediction": wandb.Image(
                    np.clip(pred_np, 0, 1), caption=f"Epoch {epoch+1} prediction"
                ),
                "val_ground_truth": wandb.Image(
                    np.clip(gt_np, 0, 1), caption=f"Epoch {epoch+1} GT"
                ),
            })

    if val_loss < best_val:
        best_val = val_loss
        best_epoch = epoch
        early_stop_counter = 0
        torch.save(model.state_dict(),
                   f'model/bestmodel/clinical_best_epoch{epoch}.pth')
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")


# ── Save loss curves ──
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('Training & Validation Loss')
ax.legend(); ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=150, bbox_inches='tight')
plt.close()


# ══════════════════════════════════════════════════════════════════════
#                       EVALUATION
# ══════════════════════════════════════════════════════════════════════

print(f"\nLoading best model (epoch {best_epoch})")
model.load_state_dict(torch.load(
    f'model/bestmodel/clinical_best_epoch{best_epoch}.pth',
    map_location=device, weights_only=True
))

test_files = [full_ds_clean.rf_files[i] for i in test_idx]
with open(os.path.join(save_dir, "test_files.txt"), 'w') as f:
    for fname in test_files:
        f.write(fname + '\n')

test_metrics = run_test(
    model, test_loader, device,
    os.path.join(save_dir, "test"),
    label="Clinical-Only Test",
    file_list=test_files
)

wandb.finish()
print(f"\nAll results saved to {save_dir}")
print("Done!")