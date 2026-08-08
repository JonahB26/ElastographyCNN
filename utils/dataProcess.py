"""
Description:
Processes ultrasound simulation data from separate folders for three
echogenicity types (hyperechoic, hypoechoic, isoechoic) and their
shared elastography labels. This data is not committed but is available upon request.

Folder layout:
    data/
      elastography labels/   →  ElastographyResult_FramePair_Result_{id}.mat
                                  with output.Elastography_Cleaned  (220×200)
      hyperechoic/           →  FramePair_Result_{id}.mat
                                  with Frames.Frame1, Frames.Frame2
      hypoechoic/            →  FramePair_Result_{id}.mat
      isoechoic/             →  FramePair_Result_{id}.mat

Each numbered .mat in the RF folders shares the same label from
elastography_labels/.  RF frames are resized to (2500, 256).

Output:
    tumor_images_final.npy                   : (N_total, 2500, 256, 2)
    tumor_labels_elastography_image.npy      : (N_total, 220, 200)

    where N_total = num_files * 3  (one entry per echogenicity type per sample)

"""

import os
import numpy as np
import h5py
from scipy import io as sio
from skimage.transform import resize as sk_resize


BASE_DIR = r"../data" 
LABEL_DIR = os.path.join(BASE_DIR, "elastography labels")
RF_DIRS = {
    "hyperechoic": os.path.join(BASE_DIR, "hyperechoic"),
    "hypoechoic":  os.path.join(BASE_DIR, "hypoechoic"),
    "isoechoic":   os.path.join(BASE_DIR, "isoechoic"),
}

TARGET_RF_SHAPE = (2500, 256)     
ELASTO_SHAPE    = (220, 200)

SAVE_DIR = "/data/train"


def resize_rf_frame(frame, target_shape=TARGET_RF_SHAPE):
    """Resize an RF frame to target_shape using bilinear interpolation.
    
    Works on float32 data; preserves the value range (no clipping to [0,1]).
    """
    return sk_resize(
        frame.astype(np.float32),
        target_shape,
        order=1,                
        preserve_range=True,     
        anti_aliasing=True,
    ).astype(np.float32)


def get_sample_ids():
    """Return sorted list of integer IDs parsed from label filenames.
    
    Expected filename pattern: ElastographyResult_FramePair_Result_{id}.mat
    """
    PREFIX = "ElastographyResult_FramePair_Result_"
    ids = []
    for f in os.listdir(LABEL_DIR):
        if f.startswith(PREFIX) and f.endswith(".mat"):

            num_str = f[len(PREFIX):-len(".mat")]
            try:
                ids.append(int(num_str))
            except ValueError:
                print(f"Skipping file with non-numeric ID: {f}")
    ids.sort()
    return ids


def load_label(sample_id):
    """Load the elastography label. Tries scipy (v5/v7) first, then h5py (v7.3)."""
    fname = f"ElastographyResult_FramePair_Result_{sample_id}.mat"
    path = os.path.join(LABEL_DIR, fname)
    try:
        mat = sio.loadmat(path)
        elasto = mat["output"][0, 0]["Elastography"].astype(np.float32) #CAN ALSO USE Elastography_Cleaned KEY FOR PROCESSED IMAGES
    except NotImplementedError:

        with h5py.File(path, 'r') as f:
            elasto = np.array(f['output']['Elastography'], dtype=np.float32).T
    assert elasto.shape == ELASTO_SHAPE, (
        f"Label {sample_id} has shape {elasto.shape}, expected {ELASTO_SHAPE}"
    )
    return elasto


def load_rf_pair(rf_dir, sample_id):
    """Load and resize Frame1/Frame2. Tries scipy (v5/v7) first, then h5py (v7.3)."""
    fname = f"FramePair_Result_{sample_id}.mat"
    path = os.path.join(rf_dir, fname)
    try:
        mat = sio.loadmat(path)
        frame1 = mat["Frames"][0, 0]["Frame1"].astype(np.float32)
        frame2 = mat["Frames"][0, 0]["Frame2"].astype(np.float32)
    except NotImplementedError:
        with h5py.File(path, 'r') as f:
            frame1 = np.array(f['Frames']['Frame1'], dtype=np.float32).T
            frame2 = np.array(f['Frames']['Frame2'], dtype=np.float32).T

    frame1 = resize_rf_frame(frame1)
    frame2 = resize_rf_frame(frame2)

    return np.stack([frame1, frame2], axis=-1)  


def create_npy_memmap(path, shape, dtype=np.float32):
    """Create a .npy file on disk with the correct header, return a writable memmap.

    This lets us write one sample at a time without holding the full array in RAM.
    """
    fp = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)
    return fp


def process_data():
    sample_ids = get_sample_ids()
    num_samples = len(sample_ids)
    num_echo_types = len(RF_DIRS)
    total = num_samples * num_echo_types

    print(f"Found {num_samples} sample IDs, {num_echo_types} echo types → {total} total entries")

    os.makedirs(SAVE_DIR, exist_ok=True)

    img_path = os.path.join(SAVE_DIR, "tumor_images_final.npy")
    lbl_path = os.path.join(SAVE_DIR, "tumor_labels_elastography_image.npy")

    images_mmap = create_npy_memmap(img_path, shape=(total, *TARGET_RF_SHAPE, 2))
    labels_mmap = create_npy_memmap(lbl_path, shape=(total, *ELASTO_SHAPE))

    idx = 0
    skipped = 0
    for sid in sample_ids:
        try:
            label = load_label(sid)
        except Exception as e:
            print(f"[WARN] Skipping label {sid} — {e}")
            skipped += num_echo_types
            continue

        for echo_name, rf_dir in RF_DIRS.items():
            try:
                rf_pair = load_rf_pair(rf_dir, sid)
            except Exception as e:
                print(f"[WARN] Skipping {echo_name}/{sid} — {e}")
                skipped += 1
                continue

            images_mmap[idx] = rf_pair
            labels_mmap[idx] = label
            idx += 1

            if idx % 100 == 0:

                images_mmap.flush()
                labels_mmap.flush()
                print(f"  [{idx}/{total}] processed ({echo_name} / {sid})")

    images_mmap.flush()
    labels_mmap.flush()

    print(f"\nDone: {idx} samples written, {skipped} skipped")

    if idx < total:
        print(f"Trimming from {total} to {idx} entries...")
        del images_mmap, labels_mmap  

        imgs = np.load(img_path, mmap_mode='r')[:idx].copy()
        lbls = np.load(lbl_path, mmap_mode='r')[:idx].copy()
        np.save(img_path, imgs)
        np.save(lbl_path, lbls)
        del imgs, lbls

    print(f"Saved to {SAVE_DIR}")


if __name__ == "__main__":
    process_data()