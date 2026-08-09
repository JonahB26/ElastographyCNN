"""
Semi-supervised training script using Mean Teacher framework.
"""


import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms.functional import resize
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sewar import vifp
from model.fullmodel import FullModel
from utils.dataloading import RFElastoDataset
from utils.clinical_dataloading import ClinicalRFDataset
from utils.balancedLoss import BalancedLoss
from utils.otherUtils import *


dataFrm1andFrm2 = "data/train/tumor_images_final.npy"
dataLabel = "data/train/tumor_labels_elastography_image.npy"
clinical_folder = "/home/deeplearningtower/Documents/TUFFC_2022_Bi_Directional_elasto"

num_epochs = 50

NUM_ECHO_TYPES = 3

pct_start_warmup = 0.1
start_lr = 3e-4
maxm_lr = 1e-3

# ── Mean Teacher hyperparameters ───────────────────────────────────────
EMA_ALPHA = 0.999           
CONSISTENCY_WEIGHT = 0.05   
RAMPUP_EPOCHS = 20           


def split_by_phantom(dataset, num_echo_types=3, ratios=(0.8, 0.1, 0.1), seed=42):
    """Split dataset so all echogenicity variants of the same phantom
    stay in the same split (train / val / test)."""
    total_samples = len(dataset)
    num_phantoms = total_samples // num_echo_types
    assert num_phantoms * num_echo_types == total_samples, (
        f"Dataset length {total_samples} is not divisible by {num_echo_types}."
    )

    rng = np.random.RandomState(seed)
    phantom_ids = np.arange(num_phantoms)
    rng.shuffle(phantom_ids)

    n_train = int(np.floor(ratios[0] * num_phantoms))
    n_val   = int(np.floor(ratios[1] * num_phantoms))

    train_phantoms = phantom_ids[:n_train]
    val_phantoms   = phantom_ids[n_train:n_train + n_val]
    test_phantoms  = phantom_ids[n_train + n_val:]

    def phantoms_to_indices(phantom_arr):
        idx = []
        for p in phantom_arr:
            start = p * num_echo_types
            idx.extend(range(start, start + num_echo_types))
        return idx

    train_idx = phantoms_to_indices(train_phantoms)
    val_idx   = phantoms_to_indices(val_phantoms)
    test_idx  = phantoms_to_indices(test_phantoms)

    print(f"Split: {len(train_phantoms)} train phantoms ({len(train_idx)} samples), "
          f"{len(val_phantoms)} val phantoms ({len(val_idx)} samples), "
          f"{len(test_phantoms)} test phantoms ({len(test_idx)} samples)")

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


@torch.no_grad()
def update_teacher(student, teacher, alpha):
    """Exponential moving average update of teacher weights."""
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(alpha).add_(s_param.data, alpha=1.0 - alpha)


def consistency_rampup(epoch, rampup_epochs):
    """Sigmoid rampup from 0 to 1 over rampup_epochs."""
    if epoch >= rampup_epochs:
        return 1.0
    phase = 1.0 - epoch / rampup_epochs
    return float(np.exp(-5.0 * phase * phase))


def augment_tensor(x):
    """Apply random augmentation to an RF tensor on GPU.
    Used to create a second view for the teacher input.
    x: (B, 2, 2500, 256)
    """
    B = x.shape[0]
    augmented = x.clone()

    for i in range(B):
        if torch.rand(1).item() < 0.5:
            augmented[i] = augmented[i].flip(-1)  

        if torch.rand(1).item() < 0.5:
            noise_std = torch.rand(1).item() * 0.02 * augmented[i].abs().max()
            augmented[i] = augmented[i] + torch.randn_like(augmented[i]) * noise_std

        if torch.rand(1).item() < 0.5:
            gain = 0.85 + torch.rand(1).item() * 0.30  
            augmented[i] = augmented[i] * gain

    for i in range(B):
        augmented[i] = augmented[i] / (augmented[i].abs().max() + 1e-8)

    return augmented


# ══════════════════════════════════════════════════════════════════════
#                           SETUP
# ══════════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Labelled synthetic data ──
train_dataset = RFElastoDataset(dataFrm1andFrm2, dataLabel, augment=True)
eval_dataset  = RFElastoDataset(dataFrm1andFrm2, dataLabel, augment=False)

train_ds, _, _ = split_by_phantom(train_dataset, NUM_ECHO_TYPES, (0.8, 0.1, 0.1), seed=42)
_, val_ds, test_ds = split_by_phantom(eval_dataset, NUM_ECHO_TYPES, (0.8, 0.1, 0.1), seed=42)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False)

# ── Unlabelled clinical data (split: 1900 train, 300 test) ──
clinical_full_ds = ClinicalRFDataset(clinical_folder, augment=False) 

num_clinical = len(clinical_full_ds)
num_clinical_test = 300
num_clinical_train = num_clinical - num_clinical_test

rng_clinical = np.random.RandomState(99)  
clinical_indices = np.arange(num_clinical)
rng_clinical.shuffle(clinical_indices)

clinical_train_idx = clinical_indices[:num_clinical_train].tolist()
clinical_test_idx  = clinical_indices[num_clinical_train:].tolist()

clinical_train_ds = Subset(clinical_full_ds, clinical_train_idx)
clinical_test_ds  = Subset(clinical_full_ds, clinical_test_idx)

clinical_loader      = DataLoader(clinical_train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
clinical_test_loader = DataLoader(clinical_test_ds,  batch_size=1, shuffle=False)

print(f"Clinical data: {num_clinical_train} for training, {num_clinical_test} for testing")

print(f"Labelled training batches: {len(train_loader)}")
print(f"Unlabelled clinical batches: {len(clinical_loader)}")

# ── Student model ──
student = FullModel()
def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
student.apply(init_weights)

# ── Teacher model ──
teacher = copy.deepcopy(student)
for param in teacher.parameters():
    param.requires_grad = False

torch.cuda.empty_cache()
student = student.to(device)
teacher = teacher.to(device)
print(f"GPU memory after model load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

criterion = BalancedLoss(
    weight_l1=1.0,
    weight_ssim=0.5,
    weight_gdl=0.5,
    weight_tv=0.01,
    weight_mean=0.1
)
consistency_criterion = nn.MSELoss()

optimizer = torch.optim.Adam(student.parameters(), lr=start_lr)

scheduler = OneCycleLR(
    optimizer,
    max_lr=maxm_lr,
    total_steps=len(train_loader) * num_epochs,
    pct_start=pct_start_warmup,
    anneal_strategy='cos',
    cycle_momentum=False
)


# ══════════════════════════════════════════════════════════════════════
#                        TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

best_loss = float('inf')
patience = 50
early_stop_counter = 0

for epoch in range(num_epochs):
    student.train()
    teacher.train()   

    train_loss_sum = 0.0
    sup_loss_sum   = 0.0
    con_loss_sum   = 0.0

    clinical_iter = iter(clinical_loader)

    w_consistency = CONSISTENCY_WEIGHT * consistency_rampup(epoch, RAMPUP_EPOCHS)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for i, (x_lab, y_lab) in enumerate(pbar):
        x_lab, y_lab = x_lab.to(device), y_lab.to(device)

        # ── 1. Supervised loss on labelled synthetic data ──
        pred_lab = student(x_lab)
        sup_loss = criterion(pred_lab, y_lab)

        # ── 2. Consistency loss on unlabelled clinical data ──
        try:
            x_clin = next(clinical_iter)
        except StopIteration:
            clinical_iter = iter(clinical_loader)
            x_clin = next(clinical_iter)

        x_clin = x_clin.to(device)

        if torch.isnan(x_clin).any() or torch.isinf(x_clin).any():
            total_loss = sup_loss
            con_loss = torch.tensor(0.0, device=device)
        else:
            # Student sees original clinical data
            pred_student = student(x_clin)

            # Teacher sees augmented version of the same clinical data
            x_clin_aug = augment_tensor(x_clin)
            with torch.no_grad():
                pred_teacher = teacher(x_clin_aug)

            con_loss = consistency_criterion(pred_student, pred_teacher.detach())

            total_loss = sup_loss + w_consistency * con_loss

        if torch.isnan(total_loss):
            print(f"[WARN] NaN loss at step {i}, skipping")
            optimizer.zero_grad()
            scheduler.step()
            continue

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5)
        optimizer.step()
        scheduler.step()

        # ── 4. Update teacher via EMA ──
        update_teacher(student, teacher, EMA_ALPHA)

        train_loss_sum += total_loss.item()
        sup_loss_sum   += sup_loss.item()
        con_loss_sum   += con_loss.item()

        # Update progress bar
        if i % 50 == 0:
            pbar.set_postfix({
                'sup': f'{sup_loss.item():.4f}',
                'con': f'{con_loss.item():.4f}',
                'w': f'{w_consistency:.3f}'
            })

    student.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = student(x)
            val_loss += criterion(pred, y).item()

        if epoch % 2 == 0:
            plot_results(pred, y, epoch, 'val')

    train_loss = train_loss_sum / len(train_loader)
    sup_avg    = sup_loss_sum / len(train_loader)
    con_avg    = con_loss_sum / len(train_loader)
    val_loss  /= len(val_loader)

    if val_loss < best_loss:
        best_loss = val_loss
        early_stop_counter = 0
        torch.save(student.state_dict(), f'model/bestmodel/best_model_epoch{epoch}.pth')
        torch.save(teacher.state_dict(), f'model/bestmodel/teacher_epoch{epoch}.pth')
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

    print(f"Epoch {epoch+1}: Total={train_loss:.4f} (sup={sup_avg:.4f}, con={con_avg:.4f}, w={w_consistency:.3f}), Val={val_loss:.4f}")


# ══════════════════════════════════════════════════════════════════════
#                          TEST
# ══════════════════════════════════════════════════════════════════════

student.eval()
test_preds = []
test_gts   = []
psnr_list  = []
ssim_list  = []
vif_list   = []
ncc_list   = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = student(x)

        pred_np = pred.squeeze().cpu().numpy()
        y_np    = y.squeeze().cpu().numpy()

        pred_np = np.clip(pred_np, 0.0, 1.0)
        y_np    = np.clip(y_np,    0.0, 1.0)

        pred_255 = (pred_np * 255).astype(np.uint8)
        y_255    = (y_np    * 255).astype(np.uint8)

        psnr_val = peak_signal_noise_ratio(y_255,   pred_255, data_range=255)
        ssim_val = structural_similarity(
            y_255, pred_255, data_range=255, channel_axis=None
        )
        vif_val  = vifp(y_255, pred_255)
        ncc_val  = compute_ncc(y_np, pred_np)

        test_preds.append(pred_np)
        test_gts.append(y_np)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        vif_list.append(vif_val)
        ncc_list.append(ncc_val)

mean_psnr = np.mean(psnr_list)
mean_ssim = np.mean(ssim_list)
std_ssim  = np.std(ssim_list)
max_ssim  = np.max(ssim_list)
min_ssim  = np.min(ssim_list)
mean_vif  = np.mean(vif_list)
mean_ncc  = np.mean(ncc_list)

print("\n=== Test Results ===")
print(f"PSNR: mean={mean_psnr:.2f} dB")
print(f"SSIM: mean={mean_ssim:.4f}, std={std_ssim:.4f}, min={min_ssim:.4f}, max={max_ssim:.4f}")
print(f"VIF : mean={mean_vif:.4f}")
print(f"NCC : mean={mean_ncc:.4f}")

np.save("results/trainresult/testSet_preds.npy", np.stack(test_preds))
np.save("results/trainresult/testSet_gts.npy",   np.stack(test_gts))
print(f"Saved {len(test_preds)} predictions and gt to .npy")

for i, (pred, gt) in enumerate(zip(test_preds, test_gts)):
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(pred, cmap='viridis')
    plt.title(f'Prediction #{i}')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(gt, cmap='viridis')
    plt.title(f'Ground Truth #{i}')
    plt.axis('off')

    plt.savefig(f"results/trainresult/images/result_{i:03d}.png", dpi=150, bbox_inches='tight')
    plt.close()

print(f"Saved {len(test_preds)} comparison images to results/trainresult/")


# ══════════════════════════════════════════════════════════════════════
#                    CLINICAL TEST (no ground truth)
# ══════════════════════════════════════════════════════════════════════

clinical_save_dir = "results/clinical_predictions/"
os.makedirs(clinical_save_dir, exist_ok=True)
os.makedirs(os.path.join(clinical_save_dir, "images"), exist_ok=True)

clinical_preds = []

student.eval()
with torch.no_grad():
    for idx, x_clin in enumerate(clinical_test_loader):
        x_clin = x_clin.to(device)
        pred = student(x_clin)

        pred_np = pred.squeeze().cpu().numpy()
        pred_np = np.clip(pred_np, 0.0, 1.0)
        clinical_preds.append(pred_np)

        plt.figure(figsize=(5, 5))
        plt.imshow(pred_np, cmap='viridis')
        plt.title(f'Clinical Prediction #{idx}')
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.savefig(f"{clinical_save_dir}/images/clinical_{idx:03d}.png", dpi=150, bbox_inches='tight')
        plt.close()

np.save(f"{clinical_save_dir}/clinical_preds.npy", np.stack(clinical_preds))

clinical_test_files = [clinical_full_ds.files[i] for i in clinical_test_idx]
with open(f"{clinical_save_dir}/clinical_test_files.txt", 'w') as f:
    for fname in clinical_test_files:
        f.write(fname + '\n')

print(f"\n=== Clinical Test ===")
print(f"Saved {len(clinical_preds)} clinical predictions to {clinical_save_dir}")
print(f"Test file list saved to {clinical_save_dir}/clinical_test_files.txt")