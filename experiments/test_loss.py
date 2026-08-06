import torch

from losses.reconstruction import ReconstructionLoss


def main():

    loss_fn = ReconstructionLoss()

    original = torch.rand(
        1, 3, 128, 128
    )

    reconstructed = torch.rand(
        1, 3, 128, 128
    )

    loss = loss_fn(
        reconstructed,
        original
    )

    print("Loss:", loss.item())


if __name__ == "__main__":
    main()