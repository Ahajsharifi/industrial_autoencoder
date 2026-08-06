import torch


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device
):

    model.train()

    total_loss = 0.0


    for images, _, _ in loader:

        images = images.to(device)


        optimizer.zero_grad()


        reconstructed = model(images)


        loss = loss_fn(
            reconstructed,
            images
        )


        loss.backward()


        optimizer.step()


        total_loss += loss.item()


    return total_loss / len(loader)