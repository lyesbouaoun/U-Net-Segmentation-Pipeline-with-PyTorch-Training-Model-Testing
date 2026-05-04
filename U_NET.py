import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.pool1=nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2=nn.MaxPool2d(2)
        self.middle = DoubleConv(128, 256)
        self.up1=nn.ConvTranspose2d(256, 128,2,stride=2)
        self.conv1=DoubleConv(256, 128)
        self.up2=nn.ConvTranspose2d(128, 64,2,stride=2)
        self.conv2=DoubleConv(128, 64)
        self.out=nn.Conv2d(64, 1,1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.pool1(x1)
        x3 = self.down2(x2)
        x4 = self.pool2(x3)
        x5 = self.middle(x4)
        x6 = self.up1(x5)
        x6 = torch.cat((x6,x3),dim=1)
        x7 = self.conv1(x6)
        x7 = self.up2(x7)
        x7 = torch.cat((x7,x1),dim=1)
        x7 = self.conv2(x7)
        return self.out(x7)
    








