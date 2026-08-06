import sys
from pathlib import Path

from data.loader import create_dataloader
import matplotlib.pyplot as plt


def main():

    loader = create_dataloader(
        root="data/mvtec",
        category="bottle",
        split="train",
        batch_size=16,
        num_workers=2
    )

    images, labels, paths = next(iter(loader))


    fig, axes = plt.subplots(
        4,
        4,
        figsize=(10, 10)
    )

    for idx, ax in enumerate(axes.flat):

        img = images[idx]

        # CHW -> HWC
        img = img.permute(1, 2, 0)

        ax.imshow(img)
        ax.axis("off")

        ax.set_title(
            f"Label: {labels[idx].item()}"
        )


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()