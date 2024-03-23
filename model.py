import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor

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


class ConvLayer(nn.Module):
    """
    A convolutional layer with batch normalization and ReLU activation.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, activation_fn=nn.ReLU()
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, kernel_size // 2
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation_fn = activation_fn

    def forward(self, x):
        return self.activation_fn(self.bn(self.conv(x)))


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, downsample_channels, out_channels, kernel_size, stride=1
    ):
        super(ConvBlock, self).__init__()
        self.conv_layers = nn.Sequential(
            ConvLayer(in_channels, downsample_channels, 1, 1),
            ConvLayer(downsample_channels, out_channels, kernel_size, stride),
        )

    def forward(self, x):
        return self.conv_layers(x)
