import torch
import numpy as np
from pathlib import Path
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
        batch_size=16,
        num_workers=0
    )


    model = AutoEncoder()


    model = load_model(
        model,
        "outputs/checkpoints/best_autoencoder.pth",
        device
    )


    scores = []
    labels = []


    for images, batch_labels, paths in loader:

        reconstructed = reconstruct(
            model,
            images,
            device
        )


        _, batch_scores = calculate_error(images.to(device),reconstructed)

        scores.extend(batch_scores.cpu().numpy())


        labels.extend(
            batch_labels.numpy()
        )


    scores = np.array(scores)
    labels = np.array(labels)

    Path("outputs").mkdir(
        exist_ok=True
    )

    np.save(
        "outputs/scores.npy",
        scores
    )

    np.save(
        "outputs/labels.npy",
        labels
    )


    print("Saved evaluation results")
    print("Scores shape:", scores.shape)
    print("Labels shape:", labels.shape)

    print("Scores:")
    print(scores)

    print()

    print("Labels:")
    print(labels)


if __name__ == "__main__":
    main()