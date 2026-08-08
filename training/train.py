"""
Just regular training using the balanced loss.
"""


import os
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
from utils.balancedLoss import BalancedLoss
from utils.otherUtils import *


dataFrm1andFrm2 = "data/train/tumor_images_final.npy"
dataLabel = "data/train/tumor_labels_elastography_image.npy"
num_epochs = 50

# dataProcess.py saves samples in order:
#   for each sample_id: [hyperechoic, hypoechoic, isoechoic]
# So indices 0,1,2 are all from phantom 0;  3,4,5 from phantom 1; etc.
NUM_ECHO_TYPES = 3   # must match the number of RF folders in dataProcess.py

pct_start_warmup = 0.1
start_lr = 3e-4
maxm_lr = 1e-3


def split_by_phantom(dataset, num_echo_types=3, ratios=(0.8, 0.1, 0.1), seed=42):
    """Split dataset so all echogenicity variants of the same phantom
    stay in the same split (train / val / test).

    Args:
        dataset:         the full RFElastoDataset
        num_echo_types:  how many consecutive entries share one phantom
        ratios:          (train, val, test) fractions — must sum to 1.0
        seed:            random seed for reproducibility

    Returns:
        train_subset, val_subset, test_subset  (torch Subset objects)
    """
    total_samples = len(dataset)
    num_phantoms = total_samples // num_echo_types
    assert num_phantoms * num_echo_types == total_samples, (
        f"Dataset length {total_samples} is not divisible by {num_echo_types}. "
        f"Check that dataProcess.py wrote exactly {num_echo_types} entries per phantom."
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
        """Expand phantom IDs to the actual dataset indices."""
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


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    for x, y in tqdm(loader, desc="Validation", leave=False):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        running_loss += criterion(pred, y).item() * x.size(0)
    return running_loss / len(loader.dataset)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = RFElastoDataset(dataFrm1andFrm2, dataLabel, augment=True)
eval_dataset  = RFElastoDataset(dataFrm1andFrm2, dataLabel, augment=False)

train_ds, _, _ = split_by_phantom(
    train_dataset,
    num_echo_types=NUM_ECHO_TYPES,
    ratios=(0.8, 0.1, 0.1),
    seed=42
)

_, val_ds, test_ds = split_by_phantom(
    eval_dataset,
    num_echo_types=NUM_ECHO_TYPES,
    ratios=(0.8, 0.1, 0.1),
    seed=42
)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)


model = FullModel().to(device)
best_loss = float('inf')
patience = 50
early_stop_counter = 0
def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
model.apply(init_weights)


criterion = BalancedLoss(
    weight_l1=1.0,
    weight_ssim=0.5,
    weight_gdl=0.5,
    weight_tv=0.01,
    weight_mean=0.1
)
optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)

scheduler = OneCycleLR(
    optimizer,
    max_lr=maxm_lr,
    total_steps= len(train_loader) * num_epochs,
    pct_start=pct_start_warmup,
    anneal_strategy='cos',
    cycle_momentum=False
)

best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for i, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()



    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            val_loss += criterion(pred, y).item()

        if epoch % 2 == 0:
            plot_results(pred, y, epoch, 'val')

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    if val_loss < best_loss:
        best_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), f'model/bestmodel/best_model_epoch{epoch}.pth')
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

model.eval()
test_preds = []
test_gts   = []
psnr_list  = []
ssim_list  = []
vif_list   = []
ncc_list   = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)

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

print(f"Saved {len(test_preds)} comparison images to ./results/trainresult/")