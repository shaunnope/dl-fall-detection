import os

import torch
from PIL import Image
from torch.utils.data import Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define dataset from yolov8 annotations
class YOLOv8Dataset(Dataset):
    """
    Dataset class for the YOLOv8 annotations.
    """

    def __init__(self, dir, transform=None, target_transform=None, device=DEVICE):
        self.img_dir = f"{dir}/images"
        self.label_dir = f"{dir}/labels"

        # extract file names from the directory, removing the file extension and parent directory
        self.img_names = [
            os.path.splitext(name)[0] for name in os.listdir(self.img_dir)
        ]

        self.transform = transform
        self.target_transform = target_transform
        self.device = device

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]

        # construct the file paths for the image and label from the directory, add extension
        img_path = os.path.join(self.img_dir, self.img_names[idx] + ".jpg")
        label_path = os.path.join(self.label_dir, self.img_names[idx] + ".txt")
        image = Image.open(img_path)
        labels = torch.tensor(
            [[float(l) for l in line.rstrip("\n").split()] for line in open(label_path)]
        )
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        return image.to(self.device), labels.to(self.device)

