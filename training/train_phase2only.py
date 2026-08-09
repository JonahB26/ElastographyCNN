"""
Phase 2 only: Load best Phase 1 checkpoint, fine-tune on clinical data.
Skips Phase 1 entirely to save time.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.filters import gaussian
from skimage.transform import resize as sk_resize
from sewar import vifp
from model.fullmodel import FullModel
from utils.clinical_dataloading import ClinicalLabelledDataset
from utils.balancedLoss import BalancedLoss
from utils.otherUtils import compute_ncc
import wandb


# ══════════════════════════════════════════════════════════════════════
#                        CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

# ── Paths ──
clinical_rf_folder    = "/home/deeplearningtower/Documents/TUFFC_2022_Bi_Directional_elasto"
clinical_label_folder = "data/RivazElastographyLabels"
clinical_mask_folder  = "data/RFLigamentMasks"
phase1_checkpoint     = "model/bestmodel/p1_matched_epoch47.pth"

# ── Phase 2 ──
PHASE2_EPOCHS     = 30
PHASE2_LR         = 3e-5
PHASE2_BATCH_SIZE = 4

# ── Splits ──
CLINICAL_TEST_COUNT = 100

# ── wandb ──
WANDB_PROJECT  = "elastography"
WANDB_RUN_NAME = "phase2_only_finetune"


# ══════════════════════════════════════════════════════════════════════
#                        UTILITIES
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(pred_np, gt_np):
    pred_np = np.clip(pred_np, 0.0, 1.0)
    gt_np   = np.clip(gt_np,   0.0, 1.0)
    p255 = (pred_np * 255).astype(np.uint8)
    g255 = (gt_np   * 255).astype(np.uint8)
    return {
        'psnr': peak_signal_noise_ratio(g255, p255, data_range=255),
        'ssim': structural_similarity(g255, p255, data_range=255, channel_axis=None),
        'vif':  vifp(g255, p255),
        'ncc':  compute_ncc(gt_np, pred_np),
    }


def print_metrics(metrics_list, label=""):
    psnr = [m['psnr'] for m in metrics_list]
    ssim = [m['ssim'] for m in metrics_list]
    vif  = [m['vif']  for m in metrics_list]
    ncc  = [m['ncc']  for m in metrics_list]
    print(f"\n=== {label} Results ({len(metrics_list)} samples) ===")
    print(f"PSNR: mean={np.mean(psnr):.2f} dB")
    print(f"SSIM: mean={np.mean(ssim):.4f}, std={np.std(ssim):.4f}, "
          f"min={np.min(ssim):.4f}, max={np.max(ssim):.4f}")
    print(f"VIF : mean={np.mean(vif):.4f}")
    print(f"NCC : mean={np.mean(ncc):.4f}")


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if torch.isnan(x).any() or torch.isinf(x).any():
                continue
            total += criterion(model(x), y).item()
            n += 1
    return total / max(n, 1)


def run_test(model, loader, device, save_dir, label="Test", log_wandb=True):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)

    model.eval()
    preds, gts, metrics = [], [], []
    wandb_images = []

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_np = pred.squeeze().cpu().numpy()
            gt_np   = y.squeeze().cpu().numpy()

            m = compute_metrics(pred_np, gt_np)
            metrics.append(m)
            preds.append(np.clip(pred_np, 0, 1))
            gts.append(np.clip(gt_np, 0, 1))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(preds[-1], cmap='viridis'); ax1.set_title(f'Prediction #{idx}'); ax1.axis('off')
            ax2.imshow(gts[-1], cmap='viridis'); ax2.set_title(f'Ground Truth #{idx}'); ax2.axis('off')
            plt.suptitle(f'SSIM={m["ssim"]:.4f}  PSNR={m["psnr"]:.1f}dB  NCC={m["ncc"]:.4f}', fontsize=11)
            img_path = os.path.join(save_dir, "images", f"result_{idx:03d}.png")
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()

            if log_wandb and idx < 50:
                wandb_images.append(wandb.Image(img_path, caption=f"#{idx} SSIM={m['ssim']:.3f}"))

    np.save(os.path.join(save_dir, "all_preds.npy"), np.stack(preds))
    np.save(os.path.join(save_dir, "all_gts.npy"), np.stack(gts))
    print_metrics(metrics, label)

    if log_wandb:
        summary = {}
        for k in ['psnr', 'ssim', 'vif', 'ncc']:
            vals = [m[k] for m in metrics]
            summary[f"{label}/{k}_mean"] = np.mean(vals)
        wandb.log(summary)
        if wandb_images:
            wandb.log({f"{label}_predictions": wandb_images})

    return metrics


# ══════════════════════════════════════════════════════════════════════
#                          SETUP
# ══════════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(
    project=WANDB_PROJECT, name=WANDB_RUN_NAME,
    config={
        "phase2_epochs": PHASE2_EPOCHS, "phase2_lr": PHASE2_LR,
        "batch_size": PHASE2_BATCH_SIZE,
        "phase1_checkpoint": phase1_checkpoint,
        "approach": "load_p1_matched + finetune_all_params",
    }
)

# ── Clinical data ──
clinical_full = ClinicalLabelledDataset(
    clinical_rf_folder, clinical_label_folder,
    mask_folder=clinical_mask_folder, augment=False
)

num_clinical = len(clinical_full)
rng_clin = np.random.RandomState(99)
clin_idx = np.arange(num_clinical)
rng_clin.shuffle(clin_idx)

clin_train_idx = clin_idx[:num_clinical - CLINICAL_TEST_COUNT].tolist()
clin_test_idx  = clin_idx[num_clinical - CLINICAL_TEST_COUNT:].tolist()

clinical_train_aug = ClinicalLabelledDataset(
    clinical_rf_folder, clinical_label_folder,
    mask_folder=clinical_mask_folder, augment=True
)
clinical_train_ds   = Subset(clinical_train_aug, clin_train_idx)
clinical_test_ds    = Subset(clinical_full, clin_test_idx)

clinical_train_loader = DataLoader(clinical_train_ds, batch_size=PHASE2_BATCH_SIZE, shuffle=True,
                                   num_workers=0, pin_memory=True)
clinical_test_loader  = DataLoader(clinical_test_ds, batch_size=1, shuffle=False, num_workers=0)

print(f"Clinical: {len(clin_train_idx)} train, {len(clin_test_idx)} test")

# ── Load Phase 1 model ──
model = FullModel()
torch.cuda.empty_cache()
model = model.to(device)
model.load_state_dict(torch.load(phase1_checkpoint, map_location=device, weights_only=True))
print(f"Loaded Phase 1 checkpoint: {phase1_checkpoint}")
print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

criterion = BalancedLoss(
    weight_l1=1.0, weight_ssim=0.5, weight_gdl=0.5,
    weight_tv=0.01, weight_mean=0.1
)

# ── Evaluate Phase 1 model on clinical BEFORE fine-tuning ──
print("\nPhase 1 model on clinical (before fine-tuning)...")
run_test(model, clinical_test_loader, device, "results/p1_on_clinical_before_ft",
         "P1 on Clinical (no FT)")


# ══════════════════════════════════════════════════════════════════════
#       PHASE 2: FINE-TUNE ON CLINICAL DATA (all params, low LR)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  PHASE 2: Fine-tuning all params on clinical data")
print("="*60)

optimizer = torch.optim.Adam(model.parameters(), lr=PHASE2_LR)
scheduler = CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS, eta_min=1e-6)

best_val_clin = float('inf')
best_epoch_clin = 0

for epoch in range(PHASE2_EPOCHS):
    model.train()
    train_loss = 0.0
    num_batches = 0

    for x, y in tqdm(clinical_train_loader, desc=f"P2 Epoch {epoch+1}"):
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
        train_loss += loss.item()
        num_batches += 1

    scheduler.step()
    train_loss /= max(num_batches, 1)
    val_loss = evaluate(model, clinical_test_loader, criterion, device)

    wandb.log({
        "epoch": epoch + 1,
        "p2/train_loss": train_loss,
        "p2/val_loss": val_loss,
        "p2/lr": optimizer.param_groups[0]['lr'],
    })

    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            for x, y in clinical_test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                break
            wandb.log({
                "p2/prediction": wandb.Image(np.clip(pred[0].squeeze().cpu().numpy(), 0, 1)),
                "p2/target": wandb.Image(np.clip(y[0].squeeze().cpu().numpy(), 0, 1)),
            })

    if val_loss < best_val_clin:
        best_val_clin = val_loss
        best_epoch_clin = epoch
        torch.save(model.state_dict(), f'model/bestmodel/p2_matched_epoch{epoch}.pth')

    print(f"P2 Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

print(f"\nLoading best Phase 2 model (epoch {best_epoch_clin})")
model.load_state_dict(torch.load(f'model/bestmodel/p2_matched_epoch{best_epoch_clin}.pth',
                                  map_location=device, weights_only=True))


# ══════════════════════════════════════════════════════════════════════
#              CLINICAL EVALUATION
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Clinical evaluation (after fine-tuning)")
print("="*60)

run_test(model, clinical_test_loader, device,
         "results/clinical_matched_results", "Clinical (matched + FT)")

clin_test_files = [clinical_full.rf_files[i] for i in clin_test_idx]
with open("results/clinical_matched_results/test_files.txt", 'w') as f:
    for fname in clin_test_files:
        f.write(fname + '\n')

wandb.finish()
print("\nDone!")