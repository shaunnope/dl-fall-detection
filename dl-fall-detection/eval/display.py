# define a library to display the images and labels, bounding boxes
# use torch, torchvision, and matplotlib

import os
import torch
import matplotlib.pyplot as plt

def display_image(image, labels, bboxes, ax=None):
    """
    Display the image with the labels and bounding boxes.

    Args:
        image (torch.Tensor): A tensor representing the image.
        labels (torch.Tensor): A tensor representing the labels.
        bboxes (torch.Tensor): A tensor representing the bounding boxes.
        ax (matplotlib.axes.Axes, optional): An axes object to plot the image. Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(image.permute(1, 2, 0))
    for label, bbox in zip(labels, bboxes):
        x, y, w, h = bbox
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", lw=2))
        ax.text(x, y, label, color="red")
    plt.show()

def display_images(images, labels, bboxes, ncols=4):
    """
    Display the images with the labels and bounding boxes.

    Args:
        images (torch.Tensor): A tensor representing the images.
        labels (torch.Tensor): A tensor representing the labels.
        bboxes (torch.Tensor): A tensor representing the bounding boxes.
        ncols (int, optional): Number of columns to display the images. Defaults to 4.
    """
    nrows = (len(images) - 1) // ncols + 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(16, 16))
    for i, (image, label, bbox) in enumerate(zip(images, labels, bboxes)):
        ax = axs[i // ncols, i % ncols]
        display_image(image, label, bbox, ax=ax)
    plt.show()

def display_prediction(image, labels, bboxes, predictions, ax=None):
    """
    Display the image with the labels, bounding boxes, and predictions.

    Args:
        image (torch.Tensor): A tensor representing the image.
        labels (torch.Tensor): A tensor representing the labels.
        bboxes (torch.Tensor): A tensor representing the bounding boxes.
        predictions (torch.Tensor): A tensor representing the predictions.
        ax (matplotlib.axes.Axes, optional): An axes object to plot the image. Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(image.permute(1, 2, 0))
    for label, bbox in zip(labels, bboxes):
        x, y, w, h = bbox
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", lw=2))
        ax.text(x, y, label, color="red")
    for prediction in predictions:
        x, y, w, h = prediction
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="blue", lw=2))
    plt.show()