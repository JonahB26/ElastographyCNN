"""
Phase 2: Progressive Unfreezing Fine-tuning on Clinical Data
=============================================================
Loads a Phase 1 checkpoint (pre-trained on synthetic) and fine-tunes on
clinical data using a 3-stage progressive unfreezing schedule:

  Stage A (epochs 1-8):   Only UNet decoder (up1-up4, outc) + final_conv trainable
  Stage B (epochs 9-18):  Unfreeze full UNet (encoder + decoder), RF encoder still frozen
  Stage C (epochs 19-30): Unfreeze everything (including RF1DEncoder)

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
from torch.optim.lr_scheduler import CosineAnnealingLR
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

# ── Paths ──
clinical_rf_folder    = "/home/deeplearningtower/Documents/TUFFC_2022_Bi_Directional_elasto"
clinical_label_folder = "data/RivazElastographyLabels"
clinical_mask_folder  = "data/RFLigamentMasks"
phase1_checkpoint     = "model/bestmodel/p1_matched_epoch47.pth"

# ── Training ──
TOTAL_EPOCHS      = 30
BATCH_SIZE        = 4
STAGE_A_END       = 8   
STAGE_B_END       = 18   

LR_STAGE_A        = 1e-4   
LR_STAGE_B        = 3e-5   
LR_STAGE_C        = 1e-5   

ANCHOR_LAMBDA     = 0.005

# ── Splits ──
CLINICAL_TEST_COUNT = 100

# ── wandb ──
WANDB_PROJECT  = "elastography"
WANDB_RUN_NAME = "phase2_progressive_unfreeze"


# ══════════════════════════════════════════════════════════════════════
#                    VERIFIED NCC IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════

def verified_ncc(pred, target):
    """
    Pearson correlation between two 2D arrays, flattened.
    Returns scalar in [-1, 1].
    """
    pred = pred.flatten().astype(np.float64)
    target = target.flatten().astype(np.float64)
    pred_c = pred - pred.mean()
    target_c = target - target.mean()
    num = np.sum(pred_c * target_c)
    den = np.sqrt(np.sum(pred_c ** 2) * np.sum(target_c ** 2))
    if den < 1e-12:
        return 0.0
    return float(num / den)


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
        'ncc_verified': verified_ncc(pred_np, gt_np),
    }


def print_metrics(metrics_list, label=""):
    print(f"\n=== {label} Results ({len(metrics_list)} samples) ===")
    for key in ['psnr', 'ssim', 'vif', 'ncc', 'ncc_verified']:
        vals = [m[key] for m in metrics_list]
        print(f"  {key:15s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
              f"min={np.min(vals):.4f}, max={np.max(vals):.4f}")

    ncc_orig = [m['ncc'] for m in metrics_list]
    ncc_veri = [m['ncc_verified'] for m in metrics_list]
    max_diff = max(abs(a - b) for a, b in zip(ncc_orig, ncc_veri))
    if max_diff > 0.01:
        print(f"\n  WARNING: NCC implementations disagree! Max diff = {max_diff:.4f}")
        print(f"       Original mean: {np.mean(ncc_orig):.4f}")
        print(f"       Verified mean: {np.mean(ncc_veri):.4f}")
    else:
        print(f"  NCC implementations agree (max diff = {max_diff:.6f})")


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


def run_test(model, loader, device, save_dir, label="Test", log_wandb=True):
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
            ax1.imshow(pred_np, cmap='viridis'); ax1.set_title(f'Prediction #{idx}'); ax1.axis('off')
            ax2.imshow(gt_np, cmap='viridis'); ax2.set_title(f'Ground Truth #{idx}'); ax2.axis('off')
            plt.suptitle(
                f'SSIM={m["ssim"]:.3f}  PSNR={m["psnr"]:.1f}dB  '
                f'NCC={m["ncc"]:.3f}  NCC_v={m["ncc_verified"]:.3f}',
                fontsize=11
            )
            img_path = os.path.join(save_dir, "images", f"result_{idx:03d}.png")
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()

            if log_wandb and idx < 50:
                wandb_images.append(wandb.Image(
                    img_path,
                    caption=f"#{idx} SSIM={m['ssim']:.3f} NCC={m['ncc']:.3f} NCC_v={m['ncc_verified']:.3f}"
                ))

    ncc_values = np.array([m['ncc'] for m in all_metrics])
    ncc_v_values = np.array([m['ncc_verified'] for m in all_metrics])
    np.save(os.path.join(save_dir, "per_sample_ncc.npy"), ncc_values)
    np.save(os.path.join(save_dir, "per_sample_ncc_verified.npy"), ncc_v_values)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ncc_values, bins=30, alpha=0.6, label=f'Original (mean={ncc_values.mean():.3f})')
    ax.hist(ncc_v_values, bins=30, alpha=0.6, label=f'Verified (mean={ncc_v_values.mean():.3f})')
    ax.set_xlabel('NCC'); ax.set_ylabel('Count'); ax.legend()
    ax.set_title(f'{label}: Per-sample NCC distribution')
    plt.savefig(os.path.join(save_dir, "ncc_histogram.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print_metrics(all_metrics, label)

    if log_wandb:
        summary = {}
        for k in ['psnr', 'ssim', 'vif', 'ncc', 'ncc_verified']:
            vals = [m[k] for m in all_metrics]
            summary[f"{label}/{k}_mean"] = np.mean(vals)
        wandb.log(summary)
        if wandb_images:
            wandb.log({f"{label}_predictions": wandb_images})

    return all_metrics


# ══════════════════════════════════════════════════════════════════════
#             PROGRESSIVE UNFREEZING HELPERS
# ══════════════════════════════════════════════════════════════════════

def freeze_module(module):
    """Freeze parameters and set BatchNorm to eval (use running stats)."""
    for param in module.parameters():
        param.requires_grad = False
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()


def unfreeze_module(module):
    """Unfreeze parameters and set BatchNorm to train."""
    for param in module.parameters():
        param.requires_grad = True
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.train()


def get_trainable_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_stage(model, stage):
    """
    Configure which parts of the model are trainable.

    Model structure (from fullmodel.py / unet.py):
        model.encoder    = RF1DEncoder  (conv1, bn1, pool1, conv2, bn2, pool2, space_conv)
        model.unet       = UNet
            .inc         = DoubleConv(64, 64)       # initial conv
            .down1       = Down(64, 128)             # encoder level 1
            .down2       = Down(128, 256)            # encoder level 2
            .down3       = Down(256, 512)            # encoder level 3
            .down4       = Down(512, 1024)           # encoder level 4
            .bottleneck  = Sequential(DoubleConv(1024,1024), Dropout)
            .up1         = Up(1024, 512)             # decoder level 1
            .up2         = Up(512, 256)              # decoder level 2
            .up3         = Up(256, 128)              # decoder level 3
            .up4         = Up(128, 64)               # decoder level 4
            .outc        = OutConv(64, 1)            # output conv
        model.final_conv = Sequential(Conv2d(1,1,3), Upsample(220,200), Tanh)
    """
    unet = model.unet

    unet_encoder = [unet.inc, unet.down1, unet.down2, unet.down3, unet.down4, unet.bottleneck]
    unet_decoder = [unet.up1, unet.up2, unet.up3, unet.up4, unet.outc]

    if stage == 'A':
        # Freeze: RF encoder + UNet encoder
        # Train:  UNet decoder + final_conv
        freeze_module(model.encoder)
        for mod in unet_encoder:
            freeze_module(mod)
        for mod in unet_decoder:
            unfreeze_module(mod)
        unfreeze_module(model.final_conv)

    elif stage == 'B':
        # Freeze: RF encoder only
        # Train:  Full UNet + final_conv
        freeze_module(model.encoder)
        for mod in unet_encoder:
            unfreeze_module(mod)
        for mod in unet_decoder:
            unfreeze_module(mod)
        unfreeze_module(model.final_conv)

    elif stage == 'C':
        # Train everything
        unfreeze_module(model.encoder)
        for mod in unet_encoder:
            unfreeze_module(mod)
        for mod in unet_decoder:
            unfreeze_module(mod)
        unfreeze_module(model.final_conv)

    trainable = get_trainable_count(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Stage {stage}: {trainable:,} / {total:,} params trainable "
          f"({100 * trainable / total:.1f}%)")


def re_freeze_for_training(model, stage):
    """
    After model.train() (which sets ALL BN to train mode),
    re-freeze the BN layers in modules that should stay frozen.
    """
    unet = model.unet
    unet_encoder = [unet.inc, unet.down1, unet.down2, unet.down3, unet.down4, unet.bottleneck]

    if stage in ('A', 'B'):
        freeze_module(model.encoder)

    if stage == 'A':
        for mod in unet_encoder:
            freeze_module(mod)


def compute_anchor_loss(model, pretrained_state, anchor_lambda):
    """L2 penalty toward pre-trained weights for trainable params only."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if param.requires_grad and name in pretrained_state:
            loss = loss + torch.sum((param - pretrained_state[name]) ** 2)
    return anchor_lambda * loss


# ══════════════════════════════════════════════════════════════════════
#                          MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=WANDB_PROJECT, name=WANDB_RUN_NAME,
        config={
            "total_epochs": TOTAL_EPOCHS,
            "stage_a_end": STAGE_A_END,
            "stage_b_end": STAGE_B_END,
            "lr_a": LR_STAGE_A, "lr_b": LR_STAGE_B, "lr_c": LR_STAGE_C,
            "anchor_lambda": ANCHOR_LAMBDA,
            "batch_size": BATCH_SIZE,
            "phase1_checkpoint": phase1_checkpoint,
            "approach": "progressive_unfreeze_with_anchor",
        }
    )

    # ── Clinical data ──
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

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    print(f"Clinical: {len(train_idx)} train, {len(test_idx)} test")

    # ── Load model ──
    model = FullModel().to(device)
    model.load_state_dict(torch.load(phase1_checkpoint, map_location=device, weights_only=True))
    print(f"Loaded checkpoint: {phase1_checkpoint}")

    pretrained_state = {}
    for name, param in model.named_parameters():
        pretrained_state[name] = param.detach().clone()

    criterion = BalancedLoss(
        weight_l1=1.0, weight_ssim=0.5, weight_gdl=0.5,
        weight_tv=0.01, weight_mean=0.1
    )

    # ── Evaluate BEFORE fine-tuning ──
    print("\n--- Phase 1 model on clinical (before fine-tuning) ---")
    run_test(model, test_loader, device, "results/progressive_before_ft",
             "P1 on Clinical (no FT)")

    # ── Training loop with progressive unfreezing ──
    best_val = float('inf')
    best_epoch = 0
    optimizer = None
    scheduler = None

    for epoch in range(1, TOTAL_EPOCHS + 1):

        # Determine current stage
        if epoch <= STAGE_A_END:
            stage = 'A'
            lr = LR_STAGE_A
        elif epoch <= STAGE_B_END:
            stage = 'B'
            lr = LR_STAGE_B
        else:
            stage = 'C'
            lr = LR_STAGE_C

        # Reconfigure at stage transitions
        if epoch == 1 or epoch == STAGE_A_END + 1 or epoch == STAGE_B_END + 1:
            print(f"\n{'=' * 60}")
            print(f"  Entering Stage {stage} at epoch {epoch}")
            print(f"{'=' * 60}")
            set_stage(model, stage)

            # New optimizer for the current set of trainable params
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=lr)

            if stage == 'A':
                stage_len = STAGE_A_END
            elif stage == 'B':
                stage_len = STAGE_B_END - STAGE_A_END
            else:
                stage_len = TOTAL_EPOCHS - STAGE_B_END
            scheduler = CosineAnnealingLR(optimizer, T_max=stage_len, eta_min=lr * 0.01)

        # ── Train one epoch ──
        model.train()
        re_freeze_for_training(model, stage)

        train_loss_sum = 0.0
        anchor_loss_sum = 0.0
        num_batches = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} (Stage {stage})"):
            x, y = x.to(device), y.to(device)
            if torch.isnan(x).any() or torch.isinf(x).any():
                continue

            optimizer.zero_grad()
            pred = model(x)
            task_loss = criterion(pred, y)
            anchor_loss = compute_anchor_loss(model, pretrained_state, ANCHOR_LAMBDA)
            loss = task_loss + anchor_loss

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            train_loss_sum += task_loss.item()
            anchor_loss_sum += anchor_loss.item()
            num_batches += 1

        scheduler.step()
        train_loss = train_loss_sum / max(num_batches, 1)
        anchor_avg = anchor_loss_sum / max(num_batches, 1)
        val_loss = evaluate(model, test_loader, criterion, device)

        wandb.log({
            "epoch": epoch,
            "stage": ord(stage) - ord('A'),
            "train_loss": train_loss,
            "anchor_loss": anchor_avg,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]['lr'],
        })

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    break
                wandb.log({
                    "prediction": wandb.Image(np.clip(pred[0].squeeze().cpu().numpy(), 0, 1)),
                    "target": wandb.Image(np.clip(y[0].squeeze().cpu().numpy(), 0, 1)),
                })

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(),
                       f'model/bestmodel/p2_progressive_epoch{epoch}.pth')

        print(f"Epoch {epoch}: Stage={stage}, Train={train_loss:.4f}, "
              f"Anchor={anchor_avg:.6f}, Val={val_loss:.4f}")

    # ── Reload best and evaluate ──
    print(f"\nLoading best model (epoch {best_epoch})")
    model.load_state_dict(torch.load(
        f'model/bestmodel/p2_progressive_epoch{best_epoch}.pth',
        map_location=device, weights_only=True
    ))

    print(f"\n{'=' * 60}")
    print("  Final Clinical Evaluation (progressive unfreezing)")
    print(f"{'=' * 60}")

    run_test(model, test_loader, device,
             "results/clinical_progressive_results",
             "Clinical (Progressive Unfreeze)")

    # Save test file list
    test_files = [clinical_full.rf_files[i] for i in test_idx]
    with open("results/clinical_progressive_results/test_files.txt", 'w') as f:
        for fname in test_files:
            f.write(fname + '\n')

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()