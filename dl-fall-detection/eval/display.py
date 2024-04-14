# define a library to display the images and labels, bounding boxes
# use torch, torchvision, and matplotlib

import os
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

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


def plot_losses(losses: dict, save_dir: str = "runs"):
    """
    Plot loss and f1 score evolution.
    """

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))

    for i, loss in enumerate(['train', 'valid']):
        for j, ltype in enumerate(['box', 'cls', 'dfl']):
            ax = axs[i, j]
            ax.plot(losses[loss]['iter'], losses[loss][ltype])
            ax.set_title(f"{loss}/{ltype}")
            ax.set_xlabel("Iterations")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # save fig
    plt.tight_layout()
    plt.savefig(f"{save_dir}/losses.png")


    f1_scores = np.array(losses['valid']['f1']).T
    fig, ax = plt.subplots(figsize=(6,4))

    for i, f1 in enumerate(f1_scores):
        ax.plot(losses['valid']['iter'], f1, label=f"class {i}")

    ax.set_title("F1 Score Evolution")
    ax.set_xlabel("Iterations")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/val_f1.png")


def plot_prc(precision, recall, threshold, save_dir = "runs"):
    nc = len(precision)

    fig, ax = plt.subplots(figsize=(6,4))

    for i in range(nc):
        ax.plot(recall[i].cpu().numpy(), precision[i].cpu().numpy(), label=f"class {i}")
    
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/pr_curve.png")

    fig, axs = plt.subplots(1, nc, figsize=(16, 4))

    for i in range(nc):
        ax = axs[i]
        ax.plot(threshold[i].cpu().numpy(), precision[i][1:].cpu().numpy(), label="Precision")
        ax.plot(threshold[i].cpu().numpy(), recall[i][1:].cpu().numpy(), label="Recall")
        ax.set_title(f"Class {i}")
        ax.set_xlabel("Threshold")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/pr_threshold.png")