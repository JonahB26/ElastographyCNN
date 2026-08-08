"""
Single-sample prediction and evaluation.

Usage:
    python predict_single.py --rf path/to/rf.mat --label path/to/label.mat

RF file:    .mat with variable 'Data' of shape (2, N, M)
Label file: .mat with variable 'reconstruction_result' of shape (220, 200), uint8 [0-255]

Saves a comparison image and prints PSNR, SSIM, VIF, NCC.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize as sk_resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sewar import vifp
from model.fullmodel import FullModel
from utils.otherUtils import compute_ncc

try:
    import h5py
except ImportError:
    h5py = None

try:
    from scipy import io as sio
except ImportError:
    sio = None


MODEL_PATH = "model/bestmodel/phase2_finetuned_epoch0.pth"  # <-- update with best model
SAVE_DIR   = "results/clinical_single_predictions"
TARGET_RF_SHAPE = (2500, 256)


def load_mat_auto(path, variable_name):
    """Load a variable from a .mat file (v5/v7 or v7.3)."""
    try:
        mat = sio.loadmat(path)
        data = mat[variable_name].astype(np.float64)
    except NotImplementedError:
        with h5py.File(path, 'r') as f:
            data = np.array(f[variable_name], dtype=np.float64).T

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    f32_max = np.finfo(np.float32).max
    data = np.clip(data, -f32_max, f32_max)
    return data.astype(np.float32)


def resize_rf_frame(frame):
    """Bilinear resize to (2500, 256)."""
    return sk_resize(
        frame, TARGET_RF_SHAPE,
        order=1, preserve_range=True, anti_aliasing=True,
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Single-sample elastography prediction")
    parser.add_argument("--rf",    required=True, help="Path to RF .mat file (variable 'Data')")
    parser.add_argument("--label", required=True, help="Path to label .mat file (variable 'reconstruction_result')")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to model checkpoint")
    parser.add_argument("--save",  default=SAVE_DIR, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.save, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullModel().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    rf_data = load_mat_auto(args.rf, 'Data')  
    frame1 = resize_rf_frame(rf_data[0])       
    frame2 = resize_rf_frame(rf_data[1])      

    img = np.stack([frame1, frame2], axis=0)   
    img = torch.from_numpy(img).unsqueeze(0).float().to(device)  
    img = img / (img.abs().max() + 1e-8)

    lbl = load_mat_auto(args.label, 'reconstruction_result') 
    gt_np = lbl.astype(np.float32) / 255.0
    gt_np = np.clip(gt_np, 0.0, 1.0)

    with torch.no_grad():
        pred = model(img)
    pred_np = pred.squeeze().cpu().numpy()
    pred_np = np.clip(pred_np, 0.0, 1.0)

    pred_255 = (pred_np * 255).astype(np.uint8)
    gt_255   = (gt_np   * 255).astype(np.uint8)

    psnr_val = peak_signal_noise_ratio(gt_255, pred_255, data_range=255)
    ssim_val = structural_similarity(gt_255, pred_255, data_range=255, channel_axis=None)
    vif_val  = vifp(gt_255, pred_255)
    ncc_val  = compute_ncc(gt_np, pred_np)

    print(f"\n=== Prediction Results ===")
    print(f"RF file:    {args.rf}")
    print(f"Label file: {args.label}")
    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"VIF : {vif_val:.4f}")
    print(f"NCC : {ncc_val:.4f}")

    rf_name = os.path.splitext(os.path.basename(args.rf))[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(pred_np, cmap='viridis')
    axes[0].set_title('Prediction')
    axes[0].axis('off')

    axes[1].imshow(gt_np, cmap='viridis')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    diff = np.abs(pred_np - gt_np)
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Absolute Difference')
    axes[2].axis('off')

    plt.suptitle(
        f'{rf_name}\nSSIM={ssim_val:.4f}  PSNR={psnr_val:.1f}dB  VIF={vif_val:.4f}  NCC={ncc_val:.4f}',
        fontsize=12
    )

    save_path = os.path.join(args.save, f"{rf_name}_prediction.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()