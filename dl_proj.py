import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import albumentations as A

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(os.path.join(root_dir, 'images'))
        self.labels = os.listdir(os.path.join(root_dir, 'labels'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'images', self.images[idx])
        label_name = os.path.join(self.root_dir, 'labels', self.labels[idx])
        
        # Load image
        image = Image.open(img_name).convert("RGB")
        image = np.asarray(image)
        
        # Load label (you need to implement this part according to your data format)
        label = self.load_label(label_name)

        # Apply transformations if specified
        if self.transform:
            label, image = self.apply_transformations(label, image)

        return image, label
    
    def apply_transformations(self, labels, image):
        transformed_labels = []

        for label in labels:
            class_label = label[0]
            x_min_ratio, y_min_ratio, bbox_width_ratio, bbox_height_ratio = label[1]

            transformed = self.transform(image=image, 
                                     bboxes=[[x_min_ratio, y_min_ratio, bbox_width_ratio, bbox_height_ratio]], 
                                     class_labels=[class_label])

            if transformed['bboxes'] != []:
                augmented_bbox = transformed['bboxes'][0]

                x_min_aug, y_min_aug, width_aug, height_aug = augmented_bbox

                if width_aug > 0 and height_aug > 0:
                    transformed_labels.append([class_label, [x_min_aug, y_min_aug, width_aug, height_aug]])
        
        augmented_image_tensor = torch.tensor(transformed['image'], dtype=torch.float32)
        return transformed_labels, augmented_image_tensor

    def load_label(self, label_path):
        with open(label_path, 'r') as label_file:
            labels = []
            for line in label_file:
                l = line.split()
                class_label = int(l[0]) - 1
                bbox_labels = [float(x) for x in l[1:]]
                if bbox_labels != []:
                    labels.append([class_label, bbox_labels])
        return labels
    
transform = A.Compose([
                A.ColorJitter(),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.5), # droped the shift scale rotate until fixed bbox issue
                ],
                bbox_params=A.BboxParams(format='yolo', min_area=0, min_visibility=0.5, label_fields=['class_labels']))

# Define paths to your dataset split folders
train_dataset = CustomDataset(root_dir='./train', transform=transform)
val_dataset = CustomDataset(root_dir='./valid', transform=transform)
test_dataset = CustomDataset(root_dir='./test', transform=transforms.ToTensor())  # No augmentation for test data

# Define data loaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=32)
# test_loader = DataLoader(test_dataset, batch_size=32)