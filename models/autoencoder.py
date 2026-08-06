import torch.nn as nn

from models.encoder import Encoder
from models.decoder import Decoder


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()


    def forward(self, x):

        latent = self.encoder(x)

        reconstructed = self.decoder(latent)

        return reconstructed