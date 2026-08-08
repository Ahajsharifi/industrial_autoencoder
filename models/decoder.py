import torch
import torch.nn as nn

from models.blocks import ConvBlock


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        # -------------------------
        # Upsampling
        # -------------------------

        self.up1 = nn.ConvTranspose2d(256, 128,kernel_size=2,stride=2)

        self.up2 = nn.ConvTranspose2d(128, 64,kernel_size=2,stride=2)

        self.up3 = nn.ConvTranspose2d(64, 32,kernel_size=2,stride=2)

        self.up4 = nn.ConvTranspose2d(32, 32,kernel_size=2,stride=2)

        # -------------------------
        # Fusion after Skip Connection
        # -------------------------

        self.conv1 = ConvBlock(384, 128)   # 128 + 256

        self.conv2 = ConvBlock(192, 64)    # 64 + 128

        self.conv3 = ConvBlock(96, 32)     # 32 + 64

        self.conv4 = ConvBlock(64, 32)     # 32 + 32

        # -------------------------
        # Final RGB reconstruction
        # -------------------------

        self.final = nn.Conv2d(32,3,kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, bottleneck, skips):

        e1, e2, e3, e4 = skips

        x = self.up1(bottleneck)
        x = torch.cat([x, e4], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, e3], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, e2], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x, e1], dim=1)
        x = self.conv4(x)

        x = self.final(x)

        return self.sigmoid(x)