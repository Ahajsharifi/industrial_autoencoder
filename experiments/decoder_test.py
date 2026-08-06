import torch

from models.encoder import Encoder
from models.decoder import Decoder


def main():

    encoder = Encoder()
    decoder = Decoder()

    x = torch.randn(1, 3, 128, 128)

    encoded = encoder(x)
    reconstructed = decoder(encoded)

    print("Input:", x.shape)
    print("Encoded:", encoded.shape)
    print("Reconstructed:", reconstructed.shape)


if __name__ == "__main__":
    main()