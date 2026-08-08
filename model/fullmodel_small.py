import torch.nn as nn
from .unet_small import UNetSmall
from .rf1DEncoder import RF1DEncoder


class FullModelSmall(nn.Module):
    """
    Smaller version of FullModel using UNetSmall (~12M params vs ~48M).
    
    Same RF1DEncoder (46K params) and final_conv — only the UNet is shrunk.
    Channel progression: 32 -> 64 -> 128 -> 256 -> 512
    vs original:         64 -> 128 -> 256 -> 512 -> 1024
    
    """

    def __init__(self):
        super().__init__()
        self.encoder = RF1DEncoder()
        self.unet = UNetSmall(n_channels=64, dropout_p=0.3)
        self.final_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Upsample(size=(220, 200), mode='bilinear', align_corners=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)     
        x = self.unet(x)        
        x = self.final_conv(x)  
        x = (x + 1) / 2         
        return x