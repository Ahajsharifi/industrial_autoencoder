import torch
from torch.utils.data import DataLoader

from data.mvtec import MVTecDataset


def create_dataloader(
    root,
    category="bottle",
    split="train",
    batch_size=32,
    shuffle=True,
    num_workers=2
):

    dataset = MVTecDataset(
        root=root,
        category=category,
        split=split
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader