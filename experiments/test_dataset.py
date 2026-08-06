from data.loader import create_dataloader


def main():

    train_loader = create_dataloader(
        root="data/mvtec",
        category="bottle",
        split="train",
        batch_size=16
    )

    images, labels, paths = next(iter(train_loader))

    print("Images:", images.shape)
    print("Labels:", labels.shape)
    print("First path:", paths[0])


if __name__ == "__main__":
    main()