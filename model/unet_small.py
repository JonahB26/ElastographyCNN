from .unet_parts import *


class UNetSmall(nn.Module):
    """
    Smaller UNet with halved channel widths.
    
    Original:  64 -> 128 -> 256 -> 512 -> 1024  (~47.9M params total)
    This:      32 ->  64 -> 128 -> 256 ->  512  (~12.0M params total)
    
    Same architecture (DoubleConv blocks, skip connections, bilinear upsampling),
    just fewer channels at every level. 
    
    Input: (B, n_channels, H, W) — n_channels=64 from RF1DEncoder (unchanged)
    Output: (B, n_classes, H, W)
    """

    def __init__(self, n_channels=64, n_classes=1, dropout_p=0.3):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder (halved channels)
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        # Bottleneck (higher dropout for smaller model)
        self.bottleneck = nn.Sequential(
            DoubleConv(512, 512),
            nn.Dropout(dropout_p)
        )

        # Decoder
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.bottleneck(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits