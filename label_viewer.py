"""
Interactive clinical label viewer with delete functionality.

Shows each clinical label as an image with Next and Delete buttons.
- Next: advances to the next image
- Delete: deletes the label .mat AND the corresponding RF .mat file, then advances

Usage:
    python label_viewer.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy import io as sio

try:
    import h5py
except ImportError:
    h5py = None


LABEL_FOLDER = "data/RivazElastographyLabels" # update path
RF_FOLDER    = "/home/deeplearningtower/Documents/TUFFC_2022_Bi_Directional_elasto" # update path
MASK_FOLDER  = "data/RFLigamentMasks"  # set to None if no mask folder


def load_label(path):
    """Load a label .mat file."""
    try:
        mat = sio.loadmat(path)
        data = mat['reconstruction_result'].astype(np.float32)
    except NotImplementedError:
        with h5py.File(path, 'r') as f:
            data = np.array(f['reconstruction_result'], dtype=np.float32).T
    return data / 255.0


class LabelViewer:
    def __init__(self, label_folder, rf_folder, mask_folder=None):
        self.label_folder = label_folder
        self.rf_folder = rf_folder
        self.mask_folder = mask_folder

        self.label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.mat')])
        self.rf_files = sorted([f for f in os.listdir(rf_folder) if f.endswith('.mat')])

        if mask_folder and os.path.exists(mask_folder):
            self.mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.mat')])
        else:
            self.mask_files = None

        self.index = 0
        self.deleted_count = 0

        print(f"Found {len(self.label_files)} labels, {len(self.rf_files)} RF files")
        if self.mask_files:
            print(f"Found {len(self.mask_files)} mask files")

        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))
        plt.subplots_adjust(bottom=0.2)

        ax_back = plt.axes([0.35, 0.05, 0.15, 0.06])
        ax_next = plt.axes([0.55, 0.05, 0.15, 0.06])
        ax_delete = plt.axes([0.75, 0.05, 0.15, 0.06])

        self.btn_back = Button(ax_back, 'Back')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_delete = Button(ax_delete, 'Delete', color='#ff6666', hovercolor='#ff3333')

        self.btn_back.on_clicked(self.on_back)
        self.btn_next.on_clicked(self.on_next)
        self.btn_delete.on_clicked(self.on_delete)

        self.show_current()
        plt.show()

    def show_current(self):
        if self.index >= len(self.label_files):
            self.ax.clear()
            self.ax.text(0.5, 0.5, f'Done!\n{self.deleted_count} files deleted\n{len(self.label_files)} remaining',
                         ha='center', va='center', fontsize=16, transform=self.ax.transAxes)
            self.ax.axis('off')
            self.fig.canvas.draw_idle()
            return

        fname = self.label_files[self.index]
        path = os.path.join(self.label_folder, fname)

        try:
            lbl = load_label(path)
            self.ax.clear()
            self.ax.imshow(lbl, cmap='viridis', vmin=0, vmax=1)
            self.ax.set_title(f'[{self.index+1}/{len(self.label_files)}] {fname}\n'
                              f'(deleted: {self.deleted_count})', fontsize=10)
            self.ax.axis('off')
        except Exception as e:
            self.ax.clear()
            self.ax.text(0.5, 0.5, f'Error loading:\n{fname}\n{e}',
                         ha='center', va='center', fontsize=10, transform=self.ax.transAxes)
            self.ax.axis('off')

        self.fig.canvas.draw_idle()

    def on_back(self, event):
        if self.index > 0:
            self.index -= 1
            self.show_current()

    def on_next(self, event):
        self.index += 1
        self.show_current()

    def on_delete(self, event):
        if self.index >= len(self.label_files):
            return

        label_fname = self.label_files[self.index]
        label_path = os.path.join(self.label_folder, label_fname)

        rf_fname = self.rf_files[self.index] if self.index < len(self.rf_files) else None
        rf_path = os.path.join(self.rf_folder, rf_fname) if rf_fname else None

        mask_fname = None
        mask_path = None
        if self.mask_files and self.index < len(self.mask_files):
            mask_fname = self.mask_files[self.index]
            mask_path = os.path.join(self.mask_folder, mask_fname)

        try:
            os.remove(label_path)
            print(f"Deleted label: {label_fname}")
        except Exception as e:
            print(f"Failed to delete label {label_fname}: {e}")

        if rf_path and os.path.exists(rf_path):
            try:
                os.remove(rf_path)
                print(f"Deleted RF:    {rf_fname}")
            except Exception as e:
                print(f"Failed to delete RF {rf_fname}: {e}")

        if mask_path and os.path.exists(mask_path):
            try:
                os.remove(mask_path)
                print(f"Deleted mask:  {mask_fname}")
            except Exception as e:
                print(f"Failed to delete mask {mask_fname}: {e}")

        self.label_files.pop(self.index)
        if rf_fname and self.index < len(self.rf_files):
            self.rf_files.pop(self.index)
        if mask_fname and self.mask_files and self.index < len(self.mask_files):
            self.mask_files.pop(self.index)

        self.deleted_count += 1

        if self.index >= len(self.label_files):
            self.index = max(0, len(self.label_files) - 1)

        self.show_current()


if __name__ == "__main__":
    LabelViewer(LABEL_FOLDER, RF_FOLDER, MASK_FOLDER)