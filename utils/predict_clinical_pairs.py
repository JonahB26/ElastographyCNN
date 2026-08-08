"""
Utility for predicting on the ClinicalData/ folder.

Loads paired _Frames/_label .mat files, injects a random ligament mask
into the RF data, runs predictions, computes metrics, saves comparison
figures, and logs to wandb.

Usage:
    from utils.predict_clinical_pairs import predict_clinical_pairs
    
    metrics = predict_clinical_pairs(
        model, device,
        clinical_folder="ClinicalData",
        mask_folder="data/RFLigamentMasks",
        save_dir="results/clinical_pairs",
        wandb_label="ClinicalData Predictions",
        log_wandb=True
    )
"""

import os
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.transform import resize as sk_resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sewar import vifp
from utils.otherUtils import compute_ncc

try:
    import h5py
except ImportError:
    h5py = None

try:
    from scipy import io as sio
except ImportError:
    sio = None

try:
    import wandb
except ImportError:
    wandb = None

TARGET_RF_SHAPE = (2500, 256)


def _load_mat_auto(path, variable_name):
    try:
        mat = sio.loadmat(path)
        data = mat[variable_name]
    except NotImplementedError:
        with h5py.File(path, 'r') as f:
            data = np.array(f[variable_name]).T
    data = np.array(data, dtype=np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    f32_max = np.finfo(np.float32).max
    return np.clip(data, -f32_max, f32_max).astype(np.float32)


def _load_mask_pair(path):
    try:
        mat = sio.loadmat(path)
        mask1 = mat['frameMasks'][0, 0]['Frame1Mask'].astype(np.float32)
        mask2 = mat['frameMasks'][0, 0]['Frame2Mask'].astype(np.float32)
    except NotImplementedError:
        with h5py.File(path, 'r') as f:
            mask1 = np.array(f['frameMasks']['Frame1Mask'], dtype=np.float32).T
            mask2 = np.array(f['frameMasks']['Frame2Mask'], dtype=np.float32).T
    mask1 = np.nan_to_num(mask1, nan=0.0, posinf=0.0, neginf=0.0)
    mask2 = np.nan_to_num(mask2, nan=0.0, posinf=0.0, neginf=0.0)
    return mask1, mask2


def _resize_rf(frame):
    return sk_resize(
        frame.astype(np.float32), TARGET_RF_SHAPE,
        order=1, preserve_range=True, anti_aliasing=True
    ).astype(np.float32)


def _compute_metrics(pred_np, gt_np):
    pred_np = np.clip(pred_np, 0.0, 1.0)
    gt_np = np.clip(gt_np, 0.0, 1.0)
    p255 = (pred_np * 255).astype(np.uint8)
    g255 = (gt_np * 255).astype(np.uint8)
    mse = np.mean((pred_np - gt_np) ** 2)
    mae = np.mean(np.abs(pred_np - gt_np))
    pred_std, gt_std = np.std(pred_np), np.std(gt_np)
    pred_range = pred_np.max() - pred_np.min()
    gt_range = gt_np.max() - gt_np.min()
    return {
        'psnr': peak_signal_noise_ratio(g255, p255, data_range=255),
        'ssim': structural_similarity(g255, p255, data_range=255, channel_axis=None),
        'vif': vifp(g255, p255),
        'ncc': compute_ncc(gt_np, pred_np),
        'mse': float(mse), 'rmse': float(np.sqrt(mse)), 'mae': float(mae),
        'contrast_ratio': float(pred_std / (gt_std + 1e-8)),
        'brightness_diff': float(np.mean(pred_np) - np.mean(gt_np)),
        'range_ratio': float(pred_range / (gt_range + 1e-8)),
    }


def _find_pairs(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith('.mat')])
    frames_files, label_files = {}, {}
    for f in files:
        if f.endswith('_Frames.mat'):
            prefix = f.replace('_Frames.mat', '')
            frames_files[prefix] = os.path.join(folder, f)
        elif f.endswith('_label.mat'):
            prefix = f.replace('_label.mat', '')
            label_files[prefix] = os.path.join(folder, f)
    pairs = []
    for prefix in sorted(frames_files.keys()):
        if prefix in label_files:
            pairs.append((prefix, frames_files[prefix], label_files[prefix]))
    return pairs


def predict_clinical_pairs(model, device, clinical_folder="ClinicalData",
                           mask_folder="data/RFLigamentMasks",
                           save_dir="results/clinical_pairs",
                           wandb_label="ClinicalData Predictions",
                           log_wandb=True):
    """
    Load paired .mat files from clinical_folder, inject a random ligament
    mask, run predictions, save comparison figures and metrics.
    
    Returns list of metric dicts.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)

    pairs = _find_pairs(clinical_folder)
    if not pairs:
        print(f"  No pairs found in {clinical_folder}/")
        return []

    mask_files = sorted(glob.glob(os.path.join(mask_folder, "*.mat")))
    if not mask_files:
        print(f"  WARNING: No mask files in {mask_folder}/ — proceeding without masks")

    print(f"  Found {len(pairs)} clinical pairs, {len(mask_files)} available masks")

    rng = np.random.RandomState(42)
    model.eval()
    all_metrics = []
    all_names = []
    wandb_images = []

    for pair_idx, (name, frames_path, label_path) in enumerate(pairs):

        data = _load_mat_auto(frames_path, 'Data')
        if data.ndim == 3 and data.shape[0] == 2:
            frame1, frame2 = data[0], data[1]
        elif data.ndim == 3 and data.shape[2] == 2:
            frame1, frame2 = data[:, :, 0], data[:, :, 1]
        else:
            print(f"    Skipping {name}: unexpected shape {data.shape}")
            continue

        frame1 = _resize_rf(frame1)
        frame2 = _resize_rf(frame2)

        if mask_files:
            mask_path = mask_files[rng.randint(len(mask_files))]
            mask1, mask2 = _load_mask_pair(mask_path)
            frame1 = frame1 + mask1
            frame2 = frame2 + mask2

        img = np.stack([frame1, frame2], axis=-1)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = img / (img.abs().max() + 1e-8)
        img = img.unsqueeze(0).to(device)

        lbl = _load_mat_auto(label_path, 'reconstruction_result')
        gt_np = np.clip(lbl.astype(np.float32) / 255.0, 0, 1)

        with torch.no_grad():
            pred = model(img)
        pred_np = np.clip(pred.squeeze().cpu().numpy(), 0, 1)

        m = _compute_metrics(pred_np, gt_np)
        all_metrics.append(m)
        all_names.append(name)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(gt_np, cmap='viridis', vmin=0, vmax=1)
        ax1.set_title('Ground Truth', fontsize=14)
        ax1.axis('off')
        ax2.imshow(pred_np, cmap='viridis', vmin=0, vmax=1)
        ax2.set_title('Prediction', fontsize=14)
        ax2.axis('off')
        plt.suptitle(f'{name}  —  NCC={m["ncc"]:.3f}  SSIM={m["ssim"]:.3f}  '
                     f'PSNR={m["psnr"]:.1f}dB  CR={m["contrast_ratio"]:.2f}', fontsize=11)
        plt.tight_layout()
        fig_path = os.path.join(save_dir, "images", f"{name}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close()

        if log_wandb and wandb is not None:
            wandb_images.append(wandb.Image(
                fig_path, caption=f"{name} NCC={m['ncc']:.3f} SSIM={m['ssim']:.3f}"))

        print(f"    {name}: NCC={m['ncc']:.3f} SSIM={m['ssim']:.3f} "
              f"PSNR={m['psnr']:.1f} MAE={m['mae']:.4f} CR={m['contrast_ratio']:.2f}")

    if all_metrics:
        print(f"\n  === {wandb_label} Summary ({len(all_metrics)} samples) ===")
        for key in ['ncc', 'ssim', 'psnr', 'mae', 'contrast_ratio']:
            vals = [m[key] for m in all_metrics]
            print(f"    {key:20s}: mean={np.mean(vals):.4f}")

        per_sample = {k: np.array([m[k] for m in all_metrics]) for k in all_metrics[0]}
        per_sample['names'] = np.array(all_names)
        np.savez(os.path.join(save_dir, "per_sample_metrics.npz"), **per_sample)

        if log_wandb and wandb is not None and wandb_images:
            wandb.log({wandb_label: wandb_images})
            summary = {}
            for k in all_metrics[0]:
                vals = [m[k] for m in all_metrics]
                summary[f"{wandb_label}/{k}_mean"] = float(np.mean(vals))
            wandb.log(summary)

    return all_metrics