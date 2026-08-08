"""
Dataset for labelled clinical RF data with ligament mask injection.

RF files:  Each .mat contains 'Data' of shape (2, N, M)
           channel 0 = pre-compression, channel 1 = post-compression
           Resized to (2500, 256).

Label files: Each .mat contains 'reconstruction_result' of shape (220, 200)
             uint8 [0, 255], normalized to [0, 1] during loading.

Mask files:  Each .mat contains struct 'frameMasks' with fields:
             'Frame1Mask' and 'Frame2Mask', both (2500, 256) single.
             These are element-wise added to the resized RF frames to
             inject simulated ligament noise into clinical data.

RF, label, and mask files are matched by sorted filename order.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from skimage.transform import resize as sk_resize

try:
    import h5py
except ImportError:
    h5py = None

try:
    from scipy import io as sio
except ImportError:
    sio = None

TARGET_RF_SHAPE = (2500, 256)
ELASTO_SHAPE = (220, 200)


def resize_rf_frame(frame, target_shape=TARGET_RF_SHAPE):
    """Bilinear resize matching MATLAB imresize behaviour."""
    return sk_resize(
        frame,
        target_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)


def load_mat_auto(path, variable_name):
    """Load a variable from a .mat file (v5/v7 or v7.3).
    Loads as float64 first to avoid overflow, then clips to float32 range.
    """
    try:
        mat = sio.loadmat(path)
        data = mat[variable_name].astype(np.float64)
    except NotImplementedError:
        with h5py.File(path, 'r') as f:
            data = np.array(f[variable_name], dtype=np.float64).T

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    f32_max = np.finfo(np.float32).max
    data = np.clip(data, -f32_max, f32_max)
    data = data.astype(np.float32)
    return data


def load_mask_pair(path):
    """Load Frame1Mask and Frame2Mask from a ligament mask .mat file.
    Returns two (2500, 256) float32 arrays.
    """
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


class ClinicalLabelledDataset(Dataset):
    """Dataset for labelled clinical RF + elastography pairs with ligament masks.

    Args:
        rf_folder:     Path to folder containing RF .mat files (variable 'Data')
        label_folder:  Path to folder containing label .mat files (variable 'reconstruction_result')
        mask_folder:   Path to folder containing ligament mask .mat files (optional, None to skip)
        augment:       Whether to apply data augmentation
    """

    def __init__(self, rf_folder: str, label_folder: str,
                 mask_folder: str = None, augment: bool = False):
        super().__init__()
        self.rf_folder = rf_folder
        self.label_folder = label_folder
        self.mask_folder = mask_folder
        self.augment = augment

        self.rf_files = sorted([f for f in os.listdir(rf_folder) if f.endswith('.mat')])
        self.label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.mat')])

        assert len(self.rf_files) == len(self.label_files), (
            f"Mismatch: {len(self.rf_files)} RF files vs {len(self.label_files)} label files"
        )

        if mask_folder is not None:
            self.mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.mat')])
            assert len(self.mask_files) == len(self.rf_files), (
                f"Mismatch: {len(self.rf_files)} RF files vs {len(self.mask_files)} mask files"
            )
            print(f"ClinicalLabelledDataset: {len(self.rf_files)} paired samples "
                  f"(with ligament masks from {mask_folder})")
        else:
            self.mask_files = None
            print(f"ClinicalLabelledDataset: {len(self.rf_files)} paired samples (no masks)")

    def __len__(self):
        return len(self.rf_files)

    def __getitem__(self, idx):
        rf_path = os.path.join(self.rf_folder, self.rf_files[idx])
        data = load_mat_auto(rf_path, 'Data') 

        frame1 = resize_rf_frame(data[0]) 
        frame2 = resize_rf_frame(data[1])  

        # Add ligament masks if available 
        if self.mask_files is not None:
            mask_path = os.path.join(self.mask_folder, self.mask_files[idx])
            mask1, mask2 = load_mask_pair(mask_path)  
            frame1 = frame1 + mask1
            frame2 = frame2 + mask2

        img = np.stack([frame1, frame2], axis=-1)  

        lbl_path = os.path.join(self.label_folder, self.label_files[idx])
        lbl = load_mat_auto(lbl_path, 'reconstruction_result')  

        if self.augment:
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :].copy()
                lbl = lbl[:, ::-1].copy()

            if np.random.rand() < 0.5:
                noise_std = np.random.uniform(0.001, 0.02) * np.abs(img).max()
                noise = np.random.randn(*img.shape).astype(np.float32) * noise_std
                img = img + noise

            if np.random.rand() < 0.5:
                gain = np.random.uniform(0.85, 1.15)
                img = img * gain

        img = torch.from_numpy(img.copy()).permute(2, 0, 1).float()  
        img = img / (img.abs().max() + 1e-8)

        lbl = torch.from_numpy(lbl.copy()).unsqueeze(0).float()  
        lbl = lbl / 255.0
        lbl = lbl.clamp(0.0, 1.0)

        return img, lbl