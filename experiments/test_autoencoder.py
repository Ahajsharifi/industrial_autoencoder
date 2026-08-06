import torch

from models.autoencoder import AutoEncoder


def main():

    model = AutoEncoder()

    x = torch.randn(
        1,
        3,
        128,
        128
    )

    output = model(x)

    print("Input:", x.shape)
    print("Output:", output.shape)


if __name__ == "__main__":
    main()