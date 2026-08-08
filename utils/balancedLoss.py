import torch
import torch.nn as nn
from piqa import SSIM

"""
This file implements a balanced loss function that combines multiple loss components to provide a more comprehensive evaluation of model performance. The loss function includes the following components:
1. L1 Loss: Measures the mean absolute error between the predicted and target values.
2. SSIM Loss: Structural Similarity Index Measure, which evaluates the similarity between the predicted and target images based on luminance, contrast, and structure.
3. Gradient Difference Loss (GDL): Measures the difference in gradients between the predicted and target images, encouraging the model to preserve edge information.
4. Total Variation Loss (TV): Encourages spatial smoothness in the predicted images by penalizing large variations between neighboring pixels.
5. Mean Loss: Measures the absolute difference in mean values between the predicted and target images, ensuring that the overall intensity levels are similar.
The weights for each loss component can be adjusted to balance their contributions to the total loss. 
"""

class BalancedLoss(nn.Module):
    def __init__(
        self,
        weight_l1: float = 1.0,
        weight_ssim: float = 0.5,
        weight_gdl: float = 0.2,
        weight_tv: float = 0.01,
        weight_mean: float = 0.1
    ):
        super().__init__()
        self.l1_loss   = nn.L1Loss()
        self.ssim_loss = SSIM(n_channels=1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.w_l1      = weight_l1
        self.w_ssim    = weight_ssim
        self.w_gdl     = weight_gdl
        self.w_tv      = weight_tv
        self.w_mean    = weight_mean

    def gradient_difference_loss(self, pred, target):
        dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dy_tgt  = target[:, :, 1:, :] - target[:, :, :-1, :]
        dx_tgt  = target[:, :, :, 1:] - target[:, :, :, :-1]
        loss_gdy = (dy_pred - dy_tgt).abs().mean()
        loss_gdx = (dx_pred - dx_tgt).abs().mean()
        return loss_gdy + loss_gdx

    def tv_loss(self, pred):
        # Total Variation Loss
        dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return (dx.abs().mean() + dy.abs().mean())

    def forward(self, pred, target):
        # 1. L1 
        loss_l1 = self.l1_loss(pred, target)

        # 2. SSIM 
        loss_ssim = 1.0 - self.ssim_loss(pred, target)

        loss_gdl = self.gradient_difference_loss(pred, target)

        loss_tv = self.tv_loss(pred)

        loss_mean = (pred.mean() - target.mean()).abs()

        total_loss = (
            self.w_l1   * loss_l1
          + self.w_ssim * loss_ssim
          + self.w_gdl  * loss_gdl
          + self.w_tv   * loss_tv
          + self.w_mean * loss_mean
        )
        return total_loss