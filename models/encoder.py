import torch.nn as nn

from models.blocks import ConvBlock


class Encoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder1 = ConvBlock(3,32)
        self.encoder2 = ConvBlock(32,64)
        self.encoder3 = ConvBlock(64,128)
        self.encoder4 = ConvBlock(128,256)
        self.pool = nn.MaxPool2d(2)



    def forward(self, x):

            e1 = self.encoder1(x)
            x = self.pool(e1)

            e2 = self.encoder2(x)
            x = self.pool(e2)

            e3 = self.encoder3(x)
            x = self.pool(e3)

            e4 = self.encoder4(x)
            x = self.pool(e4)

            bottleneck = x

            return bottleneck, [e1, e2, e3, e4]