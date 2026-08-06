import matplotlib.pyplot as plt
import torch

from data.loader import create_dataloader
from models.encoder import Encoder


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = create_dataloader(
        root="data/mvtec",
        category="bottle",
        split="train",
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = Encoder().to(device)
    model.eval()

    images, _, _ = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        features = model(images)

    print(f"Input Shape : {images.shape}")
    print(f"Feature Shape : {features.shape}")

    features = features.squeeze(0).cpu()

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):

        ax.imshow(features[i], cmap="gray")
        ax.set_title(f"Channel {i}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()