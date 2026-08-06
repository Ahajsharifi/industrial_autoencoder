from models.autoencoder import AutoEncoder


def main():
    model = AutoEncoder()

    print(model)

    total = sum(p.numel() for p in model.parameters())

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(total)
    print(trainable)


if __name__ == "__main__":
    main()