"""
This script takes the predicted and ground truth elastography images saved in .npy format and allows the user to visualize them side by 
side. The user can select a region of interest (ROI) on the prediction image to calculate 
and display the mean values of both the prediction and ground truth within that ROI.
"""


import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RectangleSelector
import matplotlib.patches as patches


preds_path = "./results/predictresult/all_preds.npy"
gts_path = "./results/predictresult/all_gts.npy"

try:
    rect_selector.disconnect_events()
    del rect_selector
except Exception:
    pass
try:
    slider.disconnect_events()
    del slider
except Exception:
    pass
plt.close('all')
preds = np.load(preds_path)
gts   = np.load(gts_path)
N, H, W = preds.shape

fig, (ax_pred, ax_gt) = plt.subplots(1, 2, figsize=(8, 4))
plt.subplots_adjust(bottom=0.25)

im_pred = ax_pred.imshow(preds[0], cmap='gray', vmin=0, vmax=1)
im_gt   = ax_gt.imshow(gts[0], cmap='gray', vmin=0, vmax=1)
ax_pred.set_title("Prediction")
ax_gt.set_title("Ground Truth")

ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, "Image Index", 0, N-1, valinit=0, valstep=1)

current_index = 0

rect_pred = patches.Rectangle((0,0), 1, 1, linewidth=2,
                              edgecolor='r', facecolor='none', visible=False)
rect_gt   = patches.Rectangle((0,0), 1, 1, linewidth=2,
                              edgecolor='r', facecolor='none', visible=False)
ax_pred.add_patch(rect_pred)
ax_gt.add_patch(rect_gt)

def update_index(val):
    global current_index
    current_index = int(val)
    im_pred.set_data(preds[current_index])
    im_gt.set_data(gts[current_index])
    rect_pred.set_visible(False)
    rect_gt.set_visible(False)
    fig.canvas.draw_idle()

slider.on_changed(update_index)

def on_select(eclick, erelease):
    if eclick.inaxes != ax_pred or erelease.inaxes != ax_pred:
        return                       

    x1 = int(round(min(eclick.xdata, erelease.xdata)))
    x2 = int(round(max(eclick.xdata, erelease.xdata)))
    y1 = int(round(min(eclick.ydata, erelease.ydata)))
    y2 = int(round(max(eclick.ydata, erelease.ydata)))

    x1, x2 = max(0, x1), min(W-1, x2)
    y1, y2 = max(0, y1), min(H-1, y2)
    w = x2 - x1 + 1
    h = y2 - y1 + 1

    for rect in (rect_pred, rect_gt):
        rect.set_xy((x1, y1))
        rect.set_width(w)
        rect.set_height(h)
        rect.set_visible(True)

    region_pred = preds[current_index][y1:y2+1, x1:x2+1]
    region_gt   = gts[current_index][y1:y2+1, x1:x2+1]
    mean_pred = region_pred.mean() if region_pred.size else float('nan')
    mean_gt   = region_gt.mean()   if region_gt.size   else float('nan')
    print(f"Image {current_index}  ROI ({x1},{y1})–({x2},{y2}) | "
          f"Pred mean: {mean_pred:.6f}  GT mean: {mean_gt:.6f}")

    fig.canvas.draw_idle()

rect_selector = RectangleSelector(
    ax_pred, on_select,
    useblit=False, button=[1],
    minspanx=1, minspany=1,
    spancoords='data', interactive=True
)

plt.show()
