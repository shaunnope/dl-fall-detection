import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def target_transform(labels: list[list[float]], device=DEVICE, seperate=False) -> tuple:
    if seperate:
        # return cls and box tensors separately
        cls_tensor = torch.tensor([label[0] for label in labels], dtype=torch.long).unsqueeze(1)
        box_tensor = torch.tensor([label[1:] for label in labels], dtype=torch.float)
        return cls_tensor.to(device), box_tensor.to(device)
    
    return torch.tensor(labels, dtype=torch.float)


def collate_fn(batch):
    device = batch[0][0].device
    # return tensor of batch ids and concatenate the images and labels
    images = []
    batch_idx = []
    labels = []

    for i, (image, label) in enumerate(batch):
        images.append(image)
        batch_idx.extend([i] * len(label))
        labels.extend(label)

    images = torch.stack(images).to(device)
    batch_idx = torch.tensor(batch_idx)
    labels = torch.stack(labels)
    combined = torch.cat([batch_idx.unsqueeze(1), labels], dim=1).to(device)
    return images, combined


# define dataset from yolov8 annotations
class YOLOv8Dataset(Dataset):
    """
    Dataset class for the YOLOv8 annotations.
    """

    def __init__(
        self, dir, transform=None, target_transform=target_transform, device=DEVICE, end=None
    ):
        if isinstance(dir, self.__class__):
            # copy the dataset
            self.img_dir = dir.img_dir
            self.label_dir = dir.label_dir
            self.img_names = dir.img_names if end is None else dir.img_names[:end]
            self.transform = dir.transform

            self.target_transform = dir.target_transform
            self.device = dir.device
            return

        self.img_dir = f"{dir}/images"
        self.label_dir = f"{dir}/labels"

        # extract file names from the directory, removing the file extension and parent directory
        self.img_names = [
            os.path.splitext(name)[0] for name in os.listdir(self.img_dir)
        ]

        if end is not None:
            self.img_names = self.img_names[:end]

        self.transform = transform
        self.target_transform = target_transform
        self.device = device

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # return tensor of batch ids and concatenate the images and labels
            samples = [self.__getitem__(i) for i in range(*idx.indices(len(self)))]
            images = torch.stack([s[0] for s in samples]) 
            batch_idx = torch.cat([
                torch.tensor(s[1].shape[0] * [i]) for i, s in enumerate(samples)
            ]).to(self.device)

            return images, batch_idx, torch.cat([s[1] for s in samples])

        # construct the file paths for the image and label from the directory, add extension
        img_path = os.path.join(self.img_dir, self.img_names[idx] + ".jpg")
        label_path = os.path.join(self.label_dir, self.img_names[idx] + ".txt")
        image = Image.open(img_path)

        with open(label_path) as f:
            labels = [
                [float(l) for l in line.rstrip("\n").split()] for line in f
            ]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels, self.device)
        return image.to(self.device), labels

def get_subset(datasets, end=8):
    return {
        "train": YOLOv8Dataset(datasets["train"], end=end**2),
        "valid": YOLOv8Dataset(datasets["valid"], end=end),
        "test": YOLOv8Dataset(datasets["test"], end=end),
    }


def get_datasets(dir, transform=None, target_transform=target_transform, device=DEVICE, end=None):
    return {
        "train": YOLOv8Dataset(f"{dir}/train", transform, target_transform, device),
        "valid": YOLOv8Dataset(f"{dir}/valid", transform, target_transform, device),
        "test": YOLOv8Dataset(f"{dir}/test", transform, target_transform, device),
    } if end is None else {
        "train": YOLOv8Dataset(f"{dir}/train", transform, target_transform, device, end**2),
        "valid": YOLOv8Dataset(f"{dir}/valid", transform, target_transform, device, end),
        "test": YOLOv8Dataset(f"{dir}/test", transform, target_transform, device, end),
    }




def get_dataloaders(datasets, batch_size=16, shuffle=True, num_workers=0):
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
        ),
        "valid": DataLoader(
            datasets["valid"],
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
        ),
    }

def get_data(dir, transform=None, target_transform=target_transform, batch_size=16, device=DEVICE, end=None):
    datasets = get_datasets(dir, transform, target_transform, device, end)
    return {
        "datasets": datasets,
        "dataloaders": get_dataloaders(datasets, batch_size=batch_size),
    }