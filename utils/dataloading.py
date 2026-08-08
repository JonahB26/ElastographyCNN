import torch
from torch.utils.data import Dataset
import numpy as np

from skimage.filters import gaussian
from skimage.transform import resize as sk_resize

from scipy.ndimage import gaussian_filter, binary_dilation, binary_closing
from skimage.restoration import inpaint

def remove_horizontal_band_artifact_inpaint(
    label,
    band_search_rows=(15, 100),
    bg_percentile=80,
    band_threshold_std=1.5,
    dilation_iters=2
):
    clean = label.astype(np.float32).copy()

    h, w = clean.shape
    r0, r1 = band_search_rows
    r0 = max(0, r0)
    r1 = min(h, r1)

    bg_cutoff = np.percentile(clean, bg_percentile)
    background_mask = clean <= bg_cutoff

    bg_vals = clean[background_mask]
    bg_mean = float(np.mean(bg_vals))
    bg_std = float(np.std(bg_vals))

    search = clean[r0:r1, :]
    threshold = bg_mean + band_threshold_std * bg_std

    band_mask_local = search > threshold
    row_fraction = band_mask_local.mean(axis=1)
    active_rows = row_fraction > 0.15

    band_mask = np.zeros_like(clean, dtype=bool)
    band_mask[r0:r1, :] = active_rows[:, None] & band_mask_local

    band_mask = binary_closing(band_mask, iterations=1)
    band_mask = binary_dilation(band_mask, iterations=dilation_iters)

    cleaned = inpaint.inpaint_biharmonic(
        clean,
        band_mask,
        channel_axis=None
    )

    cleaned = np.clip(cleaned, clean.min(), clean.max()).astype(np.float32)

    return cleaned, band_mask

def add_texture_to_inpainted_region(
    original,
    cleaned,
    band_mask,
    noise_scale=0.15,
    noise_sigma=1.5,
    blend_iters=2,
    seed=None
):
    rng = np.random.default_rng(seed)

    bg_vals = original[~band_mask]
    bg_std = float(np.std(bg_vals))

    noise = rng.normal(0, bg_std * noise_scale, size=cleaned.shape)
    noise = gaussian_filter(noise, sigma=noise_sigma)

    textured = cleaned.copy()

    textured[band_mask] += noise[band_mask]

    blend_mask = binary_dilation(band_mask, iterations=blend_iters)
    soft = gaussian_filter(textured, sigma=0.8)
    textured[blend_mask] = soft[blend_mask]

    return np.clip(textured, original.min(), original.max()).astype(np.float32)

def synth_to_clinical(label, crop_top=55, crop_bottom=160, blur_sigma=2,
                      speckle_strength=0.35, speckle_correlation=0.8,
                      output_shape=(220, 200), seed=None):
    
    rng = np.random.RandomState(seed)

    low, high = np.percentile(label, 5), np.percentile(label, 95)
    label = np.clip(label, low, high)
    label = (label - low) / (high - low + 1e-8)

    cropped = label[crop_top:crop_bottom, :]
    resized = sk_resize(cropped, output_shape, order=1, preserve_range=True).astype(np.float32)
    blurred = gaussian(resized, sigma=blur_sigma)
    raw_noise = rng.randn(*blurred.shape).astype(np.float32)
    correlated_noise = gaussian(raw_noise, sigma=speckle_correlation)
    result = blurred * (1 + speckle_strength * correlated_noise)

    return np.clip(result, 0, 1).astype(np.float32)


class RFElastoDataset(Dataset):
    def __init__(self, images_path: str, labels_path: str, augment: bool = False) -> None:
        super().__init__()

        # Memory-map the .npy files — no data is loaded into RAM until indexed
        self.images = np.load(images_path, mmap_mode='r')  # (N, 2500, 256, 2)
        self.labels = np.load(labels_path, mmap_mode='r')  # (N, 220, 200)
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # .copy() pulls just this one sample into RAM from the mmap
        img = self.images[idx].copy()   # (2500, 256, 2)
        lbl = self.labels[idx].copy()   # (220, 200)

        if self.augment:

            if np.random.rand() < 0.5:
                img = img[:, ::-1, :].copy()   
                lbl = lbl[:, ::-1].copy()      

            cleaned, mask = remove_horizontal_band_artifact_inpaint(
                lbl,
                band_search_rows=(15, 100),
                bg_percentile=80,
                band_threshold_std=1.5,
                dilation_iters=3
            )

            cleaned = add_texture_to_inpainted_region(
                original=lbl,
                cleaned=cleaned,
                band_mask=mask,
                noise_scale=0.40,
                noise_sigma=0.4,
                seed=0
            )

            cleaned = synth_to_clinical(cleaned)
            lbl = cleaned


            if np.random.rand() < 0.5:
                noise_std = np.random.uniform(0.001, 0.02) * np.abs(img).max()
                noise = np.random.randn(*img.shape).astype(np.float32) * noise_std
                img = img + noise


            if np.random.rand() < 0.5:
                gain = np.random.uniform(0.85, 1.15)
                img = img * gain

        img = torch.from_numpy(img).permute(2, 0, 1).float()  
        img = img / (img.abs().max() + 1e-8)

        lbl = torch.from_numpy(lbl).unsqueeze(0).float()      
        lbl = lbl / 255.0
        lbl = lbl.clamp(0.0, 1.0)

        return img, lbl