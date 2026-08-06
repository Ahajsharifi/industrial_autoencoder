import torch
import matplotlib.pyplot as plt

from data.loader import create_dataloader
from models.autoencoder import AutoEncoder

from engine.inference import (
    load_model,
    reconstruct,
    calculate_error
)


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    loader = create_dataloader(
        root="data/mvtec",
        category="bottle",
        split="test",
        batch_size=1,
        num_workers=0
    )


    model = AutoEncoder()

    model = load_model(
        model,
        "outputs/checkpoints/best_autoencoder.pth",
        device
    )


    images, labels, paths = next(iter(loader))


    reconstructed = reconstruct(
        model,
        images,
        device
    )


    error_map, score = calculate_error(
        images.to(device),
        reconstructed
    )


    print("Image:", paths[0])
    print("Anomaly Score:", score)


if __name__ == "__main__":
    main()