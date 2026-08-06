import torch.nn as nn

from models.blocks import ConvBlock


class Encoder(nn.Module):

    def forward(self, x):

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        bottleneck = self.pool(e4)

        return bottleneck, [e1, e2, e3, e4]