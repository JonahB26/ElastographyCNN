import torch
import torch.nn as nn
import torch.nn.functional as F

class RF1DEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(2, 32, kernel_size=48, stride=2, padding=24)
        self.bn1   = nn.BatchNorm1d(32)

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)   


        self.conv2 = nn.Conv1d(32, 32, kernel_size=24, stride=1, padding=12)
        self.bn2   = nn.BatchNorm1d(32)

        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)   
        
        self.space_conv = nn.Sequential(
          nn.Conv2d(32,64,3,padding=1),
          nn.BatchNorm2d(64),
          nn.ELU(alpha=0.1, inplace=True)
        )

    def forward(self, x):

        B, C, T, S = x.shape
        
        x = x.permute(0, 3, 1, 2).reshape(B*S, C, T)   
        
        x = F.elu(self.bn1(self.conv1(x)), alpha=0.1)
        x = self.pool1(x)                             
        
        x = F.elu(self.bn2(self.conv2(x)), alpha=0.1)
        x = self.pool2(x)                             
        
        x = x.view(B, S, 32, -1).permute(0, 2, 3, 1)    
        
        x = F.interpolate(x, size=(256,256), mode='bilinear', align_corners=False)
        return self.space_conv(x)                      