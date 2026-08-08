import torch
import matplotlib.pyplot as plt
from pathlib import Path

from data.loader import create_dataloader
from models.autoencoder import AutoEncoder
from engine.inference import load_model, reconstruct


def create_heatmap(image, reconstructed):
    """
    Calculate pixel reconstruction error map
    """

    error = torch.abs(
        image - reconstructed
    )

    heatmap = error.mean(dim=1)

    return heatmap



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
        shuffle=False,
        num_workers=0
    )


    model = AutoEncoder()

    model = load_model(
        model,
        "outputs/checkpoints/best_autoencoder.pth",
        device
    )


    model.eval()


    # گرفتن یک تصویر تست

    images, labels, paths = next(iter(loader))

    images = images.to(device)


    with torch.no_grad():

        reconstructed = reconstruct(
            model,
            images,
            device
        )


    image = images[0].cpu()
    recon = reconstructed[0].cpu()


    heatmap = create_heatmap(
        images,
        reconstructed
    )[0].cpu()



    output_dir = Path(
        "outputs/baseline"
    )

    output_dir.mkdir(
        exist_ok=True
    )


    # تبدیل CHW به HWC

    image_show = image.permute(1,2,0)

    recon_show = recon.permute(1,2,0)



    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12,4)
    )


    axes[0].imshow(
        image_show
    )

    axes[0].set_title(
        "Original"
    )


    axes[1].imshow(
        recon_show
    )

    axes[1].set_title(
        "Reconstruction"
    )


    axes[2].imshow(
        heatmap,
        cmap="hot"
    )

    axes[2].set_title(
        "Error Heatmap"
    )


    for ax in axes:
        ax.axis("off")


    plt.tight_layout()


    plt.savefig(
        output_dir / "comparison.png",
        dpi=300
    )

    plt.show()



    print(
        "Saved:",
        output_dir / "comparison.png"
    )

    print(
        "Image:",
        paths[0]
    )

    print(
        "Label:",
        labels[0].item()
    )



if __name__ == "__main__":
    main()