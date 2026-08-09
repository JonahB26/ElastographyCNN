"""
Phase 2 Only: Fine-tune on Clinical Data with Variance Reduction
=================================================================
Loads best Phase 1 checkpoint and fine-tunes on clinical data with
techniques specifically aimed at rtightening the distribution of NCC values:

  1. Online Hard Example Mining (OHEM): After warmup epochs, weight
     loss more heavily on samples the model currently gets wrong.
  2. Mixup regularization: Interpolate between training pairs to
     smooth the loss landscape and reduce overfitting.
  3. Cosine Annealing with Warm Restarts: Multiple LR cycles to
     escape local minima that ignore hard samples.
  4. NCC histogram logged to wandb every N epochs for distribution
     monitoring.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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
clinical_label_folder = "data/RivazElastographyLabels"
clinical_mask_folder  = "data/RFLigamentMasks"
phase1_checkpoint     = "model/bestmodel/p1_matched_epoch29.pth"

PHASE2_EPOCHS     = 40
PHASE2_LR         = 3e-5
PHASE2_BATCH_SIZE = 4

USE_MIXUP         = True
MIXUP_ALPHA       = 0.2       # Beta distribution parameter (lower = less mixing)
OHEM_WARMUP       = 5         # epochs of normal training before OHEM kicks in
OHEM_TOP_K        = 0.7       # fraction of hardest samples to backprop on

T_0               = 10        
T_MULT            = 1      

CLINICAL_TEST_COUNT = 100

# ── wandb ──
WANDB_PROJECT  = "elastography"
WANDB_RUN_NAME = "phase2_variance_reduction"


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
    print(f"\n=== {label} Results ({len(metrics_list)} samples) ===")
    for key in ['psnr', 'ssim', 'vif', 'ncc']:
        vals = [m[key] for m in metrics_list]
        print(f"  {key:6s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
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


def log_ncc_histogram(ncc_values, label, epoch, save_dir):
    """
    Create and log an NCC histogram centered on the mean,
    showing the +/- spread around the average.
    """
    ncc_arr = np.array(ncc_values)
    mean_ncc = np.mean(ncc_arr)
    std_ncc = np.std(ncc_arr)
    median_ncc = np.median(ncc_arr)

    fig, ax = plt.subplots(figsize=(10, 5))

    counts, bins, patches = ax.hist(ncc_arr, bins=30, alpha=0.7,
                                     color='#4C72B0', edgecolor='white')

    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < mean_ncc - std_ncc:
            patch.set_facecolor('#E74C3C') 
        elif left_edge > mean_ncc + std_ncc:
            patch.set_facecolor('#2ECC71')  
        else:
            patch.set_facecolor('#4C72B0')  

    ax.axvline(mean_ncc, color='black', linewidth=2, linestyle='-',
               label=f'Mean = {mean_ncc:.3f}')
    ax.axvline(median_ncc, color='orange', linewidth=2, linestyle='--',
               label=f'Median = {median_ncc:.3f}')
    ax.axvspan(mean_ncc - std_ncc, mean_ncc + std_ncc,
               alpha=0.15, color='gray', label=f'±1 std ({std_ncc:.3f})')
    ax.axvline(mean_ncc - 2 * std_ncc, color='gray', linewidth=1,
               linestyle=':', alpha=0.5)
    ax.axvline(mean_ncc + 2 * std_ncc, color='gray', linewidth=1,
               linestyle=':', alpha=0.5)

    ax.set_xlabel('NCC', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{label} — NCC Distribution (Epoch {epoch})\n'
                 f'Mean={mean_ncc:.3f}, Std={std_ncc:.3f}, '
                 f'Min={ncc_arr.min():.3f}, Max={ncc_arr.max():.3f}, '
                 f'<0.3: {np.sum(ncc_arr < 0.3)}, >0.8: {np.sum(ncc_arr > 0.8)}',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(max(-1, mean_ncc - 4 * std_ncc), min(1, mean_ncc + 4 * std_ncc))

    plt.tight_layout()
    hist_path = os.path.join(save_dir, f"ncc_histogram_epoch{epoch}.png")
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()

    wandb.log({
        f"{label}/ncc_histogram": wandb.Image(hist_path),
        f"{label}/ncc_mean": mean_ncc,
        f"{label}/ncc_std": std_ncc,
        f"{label}/ncc_median": median_ncc,
        f"{label}/ncc_below_0.3": int(np.sum(ncc_arr < 0.3)),
        f"{label}/ncc_above_0.8": int(np.sum(ncc_arr > 0.8)),
    })

    return hist_path


def run_test(model, loader, device, save_dir, label="Test",
             log_wandb=True, epoch=None):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)

    model.eval()
    all_metrics = []
    wandb_images = []

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_np = np.clip(pred.squeeze().cpu().numpy(), 0, 1)
            gt_np   = np.clip(y.squeeze().cpu().numpy(), 0, 1)

            m = compute_metrics(pred_np, gt_np)
            all_metrics.append(m)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(pred_np, cmap='viridis')
            ax1.set_title(f'Prediction #{idx}'); ax1.axis('off')
            ax2.imshow(gt_np, cmap='viridis')
            ax2.set_title(f'Ground Truth #{idx}'); ax2.axis('off')
            plt.suptitle(
                f'SSIM={m["ssim"]:.3f}  PSNR={m["psnr"]:.1f}dB  NCC={m["ncc"]:.3f}',
                fontsize=11
            )
            img_path = os.path.join(save_dir, "images", f"result_{idx:03d}.png")
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()

            if log_wandb and idx < 50:
                wandb_images.append(wandb.Image(
                    img_path,
                    caption=f"#{idx} SSIM={m['ssim']:.3f} NCC={m['ncc']:.3f}"
                ))

    ncc_values = [m['ncc'] for m in all_metrics]
    np.save(os.path.join(save_dir, "per_sample_ncc.npy"), np.array(ncc_values))

    print_metrics(all_metrics, label)

    if log_wandb:
        summary = {}
        for k in ['psnr', 'ssim', 'vif', 'ncc']:
            vals = [m[k] for m in all_metrics]
            summary[f"{label}/{k}_mean"] = np.mean(vals)
            summary[f"{label}/{k}_std"] = np.std(vals)
        wandb.log(summary)
        if wandb_images:
            wandb.log({f"{label}_predictions": wandb_images})

        ep = epoch if epoch is not None else 0
        log_ncc_histogram(ncc_values, label, ep, save_dir)

    return all_metrics


# ══════════════════════════════════════════════════════════════════════
#                     MIXUP HELPER
# ══════════════════════════════════════════════════════════════════════

def mixup_data(x, y, alpha=0.2):
    """
    Mixup: creates convex combinations of training pairs.
    Returns mixed inputs, mixed targets, and the mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)  
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]

    return mixed_x, mixed_y, lam


# ══════════════════════════════════════════════════════════════════════
#              OHEM (Online Hard Example Mining) LOSS
# ══════════════════════════════════════════════════════════════════════

class OHEMBalancedLoss(nn.Module):
    """
    Wraps BalancedLoss to compute per-sample losses, then backprops
    only on the hardest fraction (top_k) of samples in the batch.
    Falls back to normal loss during warmup.
    """

    def __init__(self, base_criterion, top_k=0.7):
        super().__init__()
        self.base = base_criterion
        self.top_k = top_k

    def forward(self, pred, target, use_ohem=True):
        if not use_ohem or pred.size(0) <= 1:
            return self.base(pred, target)

        per_sample_losses = []
        for i in range(pred.size(0)):
            loss_i = self.base(pred[i:i+1], target[i:i+1])
            per_sample_losses.append(loss_i)

        losses = torch.stack(per_sample_losses)

        k = max(1, int(self.top_k * len(losses)))
        topk_losses, _ = torch.topk(losses, k)

        return topk_losses.mean()


# ══════════════════════════════════════════════════════════════════════
#                          MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=WANDB_PROJECT, name=WANDB_RUN_NAME,
        config={
            "phase2_epochs": PHASE2_EPOCHS,
            "phase2_lr": PHASE2_LR,
            "batch_size": PHASE2_BATCH_SIZE,
            "phase1_checkpoint": phase1_checkpoint,
            "use_mixup": USE_MIXUP,
            "mixup_alpha": MIXUP_ALPHA,
            "ohem_warmup": OHEM_WARMUP,
            "ohem_top_k": OHEM_TOP_K,
            "t_0": T_0,
            "t_mult": T_MULT,
            "approach": "variance_reduction_ohem_mixup_warm_restarts",
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

    train_idx = idx[:num_clinical - CLINICAL_TEST_COUNT].tolist()
    test_idx  = idx[num_clinical - CLINICAL_TEST_COUNT:].tolist()

    clinical_train_aug = ClinicalLabelledDataset(
        clinical_rf_folder, clinical_label_folder,
        mask_folder=clinical_mask_folder, augment=True
    )
    train_ds = Subset(clinical_train_aug, train_idx)
    test_ds  = Subset(clinical_full, test_idx)

    train_loader = DataLoader(train_ds, batch_size=PHASE2_BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    print(f"Clinical: {len(train_idx)} train, {len(test_idx)} test")

    # ── Load Phase 1 model ──
    model = FullModel().to(device)
    model.load_state_dict(torch.load(phase1_checkpoint, map_location=device,
                                      weights_only=True))
    print(f"Loaded checkpoint: {phase1_checkpoint}")

    base_criterion = BalancedLoss(
        weight_l1=1.0, weight_ssim=0.5, weight_gdl=0.5,
        weight_tv=0.01, weight_mean=0.1
    )
    ohem_criterion = OHEMBalancedLoss(base_criterion, top_k=OHEM_TOP_K)

    # ── Evaluate BEFORE fine-tuning ──
    print("\n--- Phase 1 model on clinical (before fine-tuning) ---")
    run_test(model, test_loader, device, "results/p2_vr_before_ft",
             "P1 on Clinical (no FT)", epoch=0)

    # ── Phase 2 training ──
    print(f"\n{'=' * 60}")
    print("  PHASE 2: Fine-tuning with variance reduction")
    print(f"{'=' * 60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=PHASE2_LR)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_MULT,
                                             eta_min=1e-6)

    best_val = float('inf')
    best_epoch = 0

    for epoch in range(1, PHASE2_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        num_batches = 0
        use_ohem = (epoch > OHEM_WARMUP)

        for x, y in tqdm(train_loader, desc=f"P2 Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            if torch.isnan(x).any() or torch.isinf(x).any():
                continue

            if USE_MIXUP and epoch > 2:
                x, y, lam = mixup_data(x, y, alpha=MIXUP_ALPHA)

            optimizer.zero_grad()
            pred = model(x)
            loss = ohem_criterion(pred, y, use_ohem=use_ohem)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            train_loss_sum += loss.item()
            num_batches += 1

        scheduler.step()
        train_loss = train_loss_sum / max(num_batches, 1)
        val_loss = evaluate(model, test_loader, base_criterion, device)

        wandb.log({
            "epoch": epoch,
            "p2/train_loss": train_loss,
            "p2/val_loss": val_loss,
            "p2/lr": optimizer.param_groups[0]['lr'],
            "p2/ohem_active": int(use_ohem),
        })

        if epoch % 5 == 0 or epoch == 1:
            model.eval()

            ncc_values = []
            sample_logged = False
            with torch.no_grad():
                for x_t, y_t in test_loader:
                    x_t, y_t = x_t.to(device), y_t.to(device)
                    pred_t = model(x_t)

                    if not sample_logged:
                        wandb.log({
                            "p2/prediction": wandb.Image(
                                np.clip(pred_t[0].squeeze().cpu().numpy(), 0, 1)),
                            "p2/target": wandb.Image(
                                np.clip(y_t[0].squeeze().cpu().numpy(), 0, 1)),
                        })
                        sample_logged = True

                    pred_np = np.clip(pred_t.squeeze().cpu().numpy(), 0, 1)
                    gt_np = np.clip(y_t.squeeze().cpu().numpy(), 0, 1)
                    ncc_values.append(compute_ncc(gt_np, pred_np))

            os.makedirs("results/p2_vr_histograms", exist_ok=True)
            log_ncc_histogram(ncc_values, "p2/during_training",
                              epoch, "results/p2_vr_histograms")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(),
                       f'model/bestmodel/p2_vr_epoch{epoch}.pth')

        print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"OHEM={'ON' if use_ohem else 'off'}, "
              f"LR={optimizer.param_groups[0]['lr']:.2e}")

    print(f"\nLoading best model (epoch {best_epoch})")
    model.load_state_dict(torch.load(
        f'model/bestmodel/p2_vr_epoch{best_epoch}.pth',
        map_location=device, weights_only=True
    ))

    print(f"\n{'=' * 60}")
    print("  Final Clinical Evaluation")
    print(f"{'=' * 60}")

    final_metrics = run_test(model, test_loader, device,
                              "results/clinical_vr_results",
                              "Clinical (Variance Reduction)",
                              epoch=best_epoch)

    test_files = [clinical_full.rf_files[i] for i in test_idx]
    with open("results/clinical_vr_results/test_files.txt", 'w') as f:
        for fname in test_files:
            f.write(fname + '\n')

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()