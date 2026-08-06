import torch


def load_model(model, checkpoint_path, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    return model



def reconstruct(model, image, device):

    image = image.to(device)

    with torch.no_grad():

        reconstructed = model(image)

    return reconstructed



def calculate_error(
    original,
    reconstructed
):

    error_map = torch.abs(
        original - reconstructed
    )


    scores = (
        error_map
        .mean(dim=[1,2,3])
    )


    return error_map, scores