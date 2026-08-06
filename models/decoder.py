import torch.nn as nn

from models.blocks import ConvBlock


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(256,128,kernel_size=2,stride=2),
            ConvBlock(128, 128),

            nn.ConvTranspose2d(128,64,kernel_size=2,stride=2),
            ConvBlock(64, 64),

            nn.ConvTranspose2d(64,32,kernel_size=2,stride=2),
            ConvBlock(32, 32),

            nn.ConvTranspose2d(32,3,kernel_size=2,stride=2),

            nn.Sigmoid()
        )


    def forward(self, x):
        return self.decoder(x)