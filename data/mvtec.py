from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MVTecDataset(Dataset):
    def __init__(
        self,
        root,
        category="bottle",
        split="train",
        transform=None
    ):
        self.root = Path(root)
        self.category = category
        self.split = split

        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        self.images = []
        self.labels = []

        self._load_data()


    def _load_data(self):
        category_path = self.root / self.category

        if self.split == "train":

            good_path = category_path / "train" / "good"

            for img_path in good_path.glob("*.png"):
                self.images.append(img_path)
                self.labels.append(0)


        elif self.split == "test":

            test_path = category_path / "test"

            classes = [
                "good",
                "broken_large",
                "broken_small",
                "contamination"
            ]

            for cls in classes:

                cls_path = test_path / cls

                label = 0 if cls == "good" else 1

                for img_path in cls_path.glob("*.png"):
                    self.images.append(img_path)
                    self.labels.append(label)


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):

        img_path = self.images[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label, str(img_path)