import torch.nn as nn


class ConvLayer(nn.Module):
    """
    A convolutional layer with batch normalization and ReLU activation.
    """

    # NOTE: Similfied implementation of Conv layers from YOLOv8
    # NOTE: Original default_act is nn.SiLU() but nn.ReLU() is used here
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


class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ResidualConv, self).__init__()
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else ConvLayer(in_channels, out_channels, 1, stride)
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
