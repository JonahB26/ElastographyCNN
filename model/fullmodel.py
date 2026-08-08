import torch.nn as nn
from .unet import UNet
from .rf1DEncoder import RF1DEncoder

class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RF1DEncoder()
        self.unet = UNet(n_channels=64)
        self.final_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Upsample(size=(220,200), mode='bilinear', align_corners=False),
            #nn.Sigmoid()
            nn.Tanh()
        )

    
    def forward(self, x):
            x = self.encoder(x)     
            x = self.unet(x)        
            x = self.final_conv(x)  
            x = (x + 1) / 2         
            return x