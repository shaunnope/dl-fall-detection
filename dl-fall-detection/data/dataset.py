import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import albumentations as A

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_data = A.Compose([
            A.ColorJitter(),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.5), # droped the shift scale rotate until fixed bbox issue
            A.Normalize()
            ],
            bbox_params=A.BboxParams(format='yolo', min_area=0, min_visibility=0.5, label_fields=['class_labels']))

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
    batch_idx = torch.tensor(batch_idx).to(device)
    labels = torch.stack(labels).to(device)
    combined = torch.cat([batch_idx.unsqueeze(1), labels], dim=1).to(device)
    return images, combined


# define dataset from yolov8 annotations
class YOLOv8Dataset(Dataset):
    """
    Dataset class for the YOLOv8 annotations.
    """

    def __init__(
        self, dir, transform=transform_data, device=DEVICE, end=None
    ):
        if isinstance(dir, self.__class__):
            # copy the dataset
            self.img_dir = dir.img_dir
            self.label_dir = dir.label_dir
            self.img_names = dir.img_names if end is None else dir.img_names[:end]
            self.transform = dir.transform
            #removed target_transform
            self.device = dir.device
            return

        self.save_dir = dir
        self.img_dir = f"{dir}/images"
        self.label_dir = f"{dir}/labels"

        # extract file names from the directory, removing the file extension and parent directory
        self.img_names = [
            os.path.splitext(name)[0] for name in os.listdir(self.img_dir)
        ]

        if end is not None:
            self.img_names = self.img_names[:end]

        self.transform = transform
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
        image = np.asarray(image)

        labels = self.load_label(label_path)

        if self.transform:
            try:
                labels, image = self.apply_transformations(labels, image)
                image = torch.tensor(image, dtype=torch.float32).to(self.device).permute(2, 0, 1)
                labels = torch.tensor(labels, dtype=torch.float32).to(self.device)
            except ValueError as e:
                print(idx, self.img_names[idx])
                print(labels)
                raise e
        return image, labels
    
    def apply_transformations(self, labels, image):
        bboxes = [[label[1][0], label[1][1], label[1][2], label[1][3]] for label in labels]
        class_labels = [label[0] for label in labels]

        # Apply transformations to image and bounding boxes
        transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)

        transformed_labels = []
        for class_label, bbox_label in zip(transformed['class_labels'], transformed['bboxes']):
            transformed_labels.append([class_label, *bbox_label])

        augmented_image_tensor = torch.tensor(transformed['image'], dtype=torch.float32)
        return transformed_labels, augmented_image_tensor
    
    
    def load_label(self, label_path):
        with open(label_path, 'r') as label_file:
            labels = []
            for line in label_file:
                l = line.split()
                class_label = int(l[0])
                bbox_labels = [float(x) for x in l[1:]]
                if bbox_labels != []:
                    labels.append([class_label, bbox_labels])
        return labels
        
    def relabel(self, label_dict, dir = "relabeled"):
        assert isinstance(dir, str), "dir must be a string"

        save_dir = f"{self.label_dir}_{dir}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for img_name in self.img_names:
            label_path = os.path.join(self.label_dir, img_name + ".txt")
            with open(label_path) as f:
                labels = [
                    [float(l) for l in line.rstrip("\n").split()] for line in f
                ]
            with open(os.path.join(save_dir, img_name + ".txt"), "w") as f:
                for label in labels:
                    if label[0] in label_dict:
                        label[0] = label_dict[label[0]]
                        f.write(" ".join(map(str, label)) + "\n")


def get_subset(datasets, end=8):
    return {
        "train": YOLOv8Dataset(datasets["train"], end=end**2),
        "valid": YOLOv8Dataset(datasets["valid"], end=end),
        "test": YOLOv8Dataset(datasets["test"], end=end),
    }


def get_datasets(dir, transform=None, device=DEVICE, end=None):
    return {
        "train": YOLOv8Dataset(f"{dir}/train", transform, device),
        "valid": YOLOv8Dataset(f"{dir}/valid", transform, device),
        "test": YOLOv8Dataset(f"{dir}/test", transform, device),
    } if end is None else {
        "train": YOLOv8Dataset(f"{dir}/train", transform, device, end**2),
        "valid": YOLOv8Dataset(f"{dir}/valid", transform, device, end),
        "test": YOLOv8Dataset(f"{dir}/test", transform, device, end),
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

def get_data(dir, transform=None, batch_size=16, device=DEVICE, end=None):
    datasets = get_datasets(dir, transform, device, end)
    return {
        "datasets": datasets,
        "dataloaders": get_dataloaders(datasets, batch_size=batch_size),
    }