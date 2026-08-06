import torch
import matplotlib.pyplot as plt

from data.loader import create_dataloader
from models.autoencoder import AutoEncoder


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    loader = create_dataloader(
        root="data/mvtec",
        category="bottle",
        split="train",
        batch_size=1,
        num_workers=0
    )


    model = AutoEncoder().to(device)


    checkpoint = torch.load(
        "outputs/checkpoints/best_autoencoder.pth",
        map_location=device
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model.eval()


    image, _, path = next(iter(loader))

    image = image.to(device)


    with torch.no_grad():

        reconstruction = model(image)


    image = image.cpu().squeeze(0).permute(1,2,0)

    reconstruction = (
        reconstruction
        .cpu()
        .squeeze(0)
        .permute(1,2,0)
    )


    plt.figure(figsize=(8,4))


    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")


    plt.subplot(1,2,2)
    plt.imshow(reconstruction)
    plt.title("Reconstruction")
    plt.axis("off")


    plt.suptitle(path[0])

    plt.show()



if __name__ == "__main__":
    main()