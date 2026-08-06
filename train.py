import torch
from pathlib import Path


from data.loader import create_dataloader
from models.autoencoder import AutoEncoder
from losses.reconstruction import ReconstructionLoss

from engine.train import train_one_epoch



def main():
    losses = []

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)


    # Dataset

    train_loader = create_dataloader(
        root="data/mvtec",
        category="bottle",
        split="train",
        batch_size=16,
        num_workers=2
    )


    # Model

    model = AutoEncoder().to(device)


    # Loss

    loss_fn = ReconstructionLoss()


    # Optimizer

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)


    epochs = 50


    best_loss = float("inf")


    checkpoint_dir = Path(
        "outputs/checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    for epoch in range(epochs):

        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device
        )
        losses.append(loss)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {loss:.6f}"
        )


        # save best model

        if loss < best_loss:

            best_loss = loss


            torch.save(
                {
                    "epoch": epoch + 1,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "loss":
                        loss
                },

                checkpoint_dir /
                "best_autoencoder.pth"
            )


            print("Checkpoint saved!")


if __name__ == "__main__":
    main()