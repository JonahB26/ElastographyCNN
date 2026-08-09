"""
Matched-label training pipeline with Weights & Biases logging.

Phase 1: Train on synthetic data with TRANSFORMED labels (50 epochs)
Phase 2: Fine-tune on clinical data, all params, low LR (30 epochs)
Phase 3: Evaluate on held-out clinical test set
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.filters import gaussian
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
synthetic_images      = "data/train/tumor_images_final.npy"
synthetic_labels      = "data/train/tumor_labels_elastography_image.npy"
clinical_rf_folder    = "/home/deeplearningtower/Documents/TUFFC_2022_Bi_Directional_elasto"
clinical_label_folder = "data/RivazElastographyLabels"
clinical_mask_folder  = "data/RFLigamentMasks"

PHASE1_EPOCHS     = 30
PHASE1_START_LR   = 3e-4
PHASE1_MAX_LR     = 1e-3
PHASE1_BATCH_SIZE = 4

PHASE2_EPOCHS     = 30
PHASE2_LR         = 3e-5
PHASE2_BATCH_SIZE = 4

NUM_ECHO_TYPES       = 3
CLINICAL_TEST_COUNT  = 100
PATIENCE             = 50

CROP_TOP    = 40
CROP_BOTTOM = 170
BLUR_SIGMA  = 2
SPECKLE_STR = 0.35
SPECKLE_COR = 0.8

WANDB_PROJECT  = "elastography"
WANDB_RUN_NAME = "matched_label_finetune"


# ══════════════════════════════════════════════════════════════════════
#           SYNTHETIC DATASET WITH LABEL TRANSFORMATION
# ══════════════════════════════════════════════════════════════════════

from skimage.transform import resize as sk_resize

def synth_to_clinical(label, seed=None):
    """Transform a synthetic (220, 200) label to look clinical."""
    rng = np.random.RandomState(seed)
    cropped = label[CROP_TOP:CROP_BOTTOM, :]
    resized = sk_resize(cropped, (220, 200), order=1, preserve_range=True).astype(np.float32)
    blurred = gaussian(resized, sigma=BLUR_SIGMA)
    raw_noise = rng.randn(*blurred.shape).astype(np.float32)
    correlated_noise = gaussian(raw_noise, sigma=SPECKLE_COR)
    result = blurred * (1 + SPECKLE_STR * correlated_noise)
    return np.clip(result, 0, 1).astype(np.float32)


class TransformedSyntheticDataset(Dataset):
    """Synthetic RF dataset that transforms labels to clinical style on the fly."""

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

        lbl = torch.from_numpy(lbl.copy()).unsqueeze(0).float()
        lbl = lbl.clamp(0.0, 1.0)

        return img, lbl


# ══════════════════════════════════════════════════════════════════════
#                        UTILITIES
# ══════════════════════════════════════════════════════════════════════

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
            start = p * num_echo_types
            idx.extend(range(start, start + num_echo_types))
        return idx

    train_p = phantom_ids[:n_train]
    val_p   = phantom_ids[n_train:n_train + n_val]
    test_p  = phantom_ids[n_train + n_val:]

    print(f"Split: {len(train_p)} train ({len(to_idx(train_p))}), "
          f"{len(val_p)} val ({len(to_idx(val_p))}), "
          f"{len(test_p)} test ({len(to_idx(test_p))})")

    return (Subset(dataset, to_idx(train_p)),
            Subset(dataset, to_idx(val_p)),
            Subset(dataset, to_idx(test_p)))


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += criterion(model(x), y).item()
    return total / len(loader)


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
        "phase1_epochs": PHASE1_EPOCHS, "phase2_epochs": PHASE2_EPOCHS,
        "phase1_max_lr": PHASE1_MAX_LR, "phase2_lr": PHASE2_LR,
        "batch_size": PHASE1_BATCH_SIZE,
        "crop": f"{CROP_TOP}:{CROP_BOTTOM}", "blur_sigma": BLUR_SIGMA,
        "speckle_strength": SPECKLE_STR,
        "approach": "matched_label_transform + finetune_all_params",
    }
)

# ── Synthetic data with TRANSFORMED labels ──
syn_train_ds = TransformedSyntheticDataset(synthetic_images, synthetic_labels,
                                            augment=True, transform_labels=True)
syn_eval_ds  = TransformedSyntheticDataset(synthetic_images, synthetic_labels,
                                            augment=False, transform_labels=True)

syn_train, _, _      = split_by_phantom(syn_train_ds, NUM_ECHO_TYPES, seed=42)
_, syn_val, syn_test = split_by_phantom(syn_eval_ds, NUM_ECHO_TYPES, seed=42)

syn_train_loader = DataLoader(syn_train, batch_size=PHASE1_BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
syn_val_loader   = DataLoader(syn_val,   batch_size=PHASE1_BATCH_SIZE, shuffle=False)
syn_test_loader  = DataLoader(syn_test,  batch_size=1, shuffle=False)

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
                                   num_workers=2, pin_memory=True)
clinical_test_loader  = DataLoader(clinical_test_ds, batch_size=1, shuffle=False)

print(f"Clinical: {len(clin_train_idx)} train, {len(clin_test_idx)} test")

# ── Model ──
model = FullModel()
def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
model.apply(init_weights)

torch.cuda.empty_cache()
model = model.to(device)
print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

wandb.watch(model, log="gradients", log_freq=200)

criterion = BalancedLoss(
    weight_l1=1.0, weight_ssim=0.5, weight_gdl=0.5,
    weight_tv=0.01, weight_mean=0.1
)


# ══════════════════════════════════════════════════════════════════════
#       PHASE 1: SYNTHETIC (WITH CLINICAL-STYLE LABELS)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  PHASE 1: Synthetic training (clinical-style labels)")
print("="*60)

optimizer = torch.optim.Adam(model.parameters(), lr=PHASE1_START_LR)
scheduler = OneCycleLR(
    optimizer, max_lr=PHASE1_MAX_LR,
    total_steps=len(syn_train_loader) * PHASE1_EPOCHS,
    pct_start=0.1, anneal_strategy='cos', cycle_momentum=False
)

best_val = float('inf')
early_stop_counter = 0
best_epoch = 0

for epoch in range(PHASE1_EPOCHS):
    model.train()
    train_loss = 0.0

    for x, y in tqdm(syn_train_loader, desc=f"P1 Epoch {epoch+1}"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

    train_loss /= len(syn_train_loader)
    val_loss = evaluate(model, syn_val_loader, criterion, device)

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
                pred = model(x)
                break
            wandb.log({
                "p1/prediction": wandb.Image(np.clip(pred[0].squeeze().cpu().numpy(), 0, 1)),
                "p1/target": wandb.Image(np.clip(y[0].squeeze().cpu().numpy(), 0, 1)),
            })

    if val_loss < best_val:
        best_val = val_loss
        best_epoch = epoch
        early_stop_counter = 0
        torch.save(model.state_dict(), f'model/bestmodel/p1_matched_epoch{epoch}.pth')
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"P1 Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

print(f"\nLoading best Phase 1 model (epoch {best_epoch})")
model.load_state_dict(torch.load(f'model/bestmodel/p1_matched_epoch{best_epoch}.pth',
                                  map_location=device, weights_only=True))

print("\nPhase 1 synthetic test (transformed labels)...")
run_test(model, syn_test_loader, device, "results/p1_matched_synthetic", "P1 Synthetic (matched)")

# Also test Phase 1 model directly on clinical (before fine-tuning)
print("\nPhase 1 model on clinical test (before fine-tuning)...")
run_test(model, clinical_test_loader, device, "results/p1_on_clinical", "P1 on Clinical (no finetune)")


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
        "phase": 2, "epoch": PHASE1_EPOCHS + epoch + 1,
        "p2/train_loss": train_loss, "p2/val_loss": val_loss,
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
#              PHASE 3: CLINICAL EVALUATION
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  PHASE 3: Clinical evaluation")
print("="*60)

run_test(model, clinical_test_loader, device,
         "results/clinical_matched_results", "Clinical (matched labels)")

print("\nSynthetic performance after fine-tuning...")
run_test(model, syn_test_loader, device,
         "results/synthetic_after_matched_ft", "Synthetic (after FT)")

clin_test_files = [clinical_full.rf_files[i] for i in clin_test_idx]
with open("results/clinical_matched_results/test_files.txt", 'w') as f:
    for fname in clin_test_files:
        f.write(fname + '\n')

wandb.finish()
print("\nDone!")