import torch

from models.encoder import Encoder


def main():

    model = Encoder()
    model.eval()

    x = torch.randn(1, 3, 128, 128)

    with torch.no_grad():
        bottleneck, skips = model(x)

    print("Bottleneck:", bottleneck.shape)

    shape_list = [
        (1, 32, 128, 128),
        (1, 64, 64, 64),
        (1, 128, 32, 32),
        (1, 256, 16, 16)
    ]

    assert bottleneck.shape == (1, 256, 8, 8)

    for i, skip in enumerate(skips):
        print(f"e{i+1}: {skip.shape}")

        assert skip.shape == shape_list[i], (
            f"Expected {shape_list[i]}, got {skip.shape}"
        )

    print("Encoder test passed")


if __name__ == "__main__":
    main()