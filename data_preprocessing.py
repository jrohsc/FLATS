import numpy as np
import torch 
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import os

def preprocessing(main_folder_path):
    # list of folders in the main folder
    listdir = sorted(os.listdir(main_folder_path))
    
    # Image Transformation
    mean = [0.485, 0.456, 0.406] # mean of image tensor
    std = [0.229, 0.224, 0.225]  # standard devisation of image tensor
    normalize_stat = (mean, std)
    size = 224

    transform = T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(*normalize_stat)
    ])

    dataset = ImageFolder(main_folder_path, transform=transform)
    classes = dataset.classes

    return dataset, classes

def dataloader(dataset):
    torch.manual_seed(42)
    num_val = int(len(dataset) * 0.1)

    # 1. Train and Validation Data
    train_data, val_data = random_split(dataset, [len(dataset) - num_val, num_val])

    # 2. Train and Validation Dataloader
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=3)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=3)

    return train_loader, val_loader
    

# Show single image
def imshow(image, label, dataset):
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    mean = [0.485, 0.456, 0.406] # mean of image tensor
    std = [0.229, 0.224, 0.225]  # standard devisation of image tensor

    image = image * std + mean
    image = np.clip(image, 0, 1)

    plt.imshow(image)
    plt.title(dataset.classes[label])
    plt.axis("off")

# Show batch images
def batch_imshow(dataloader):
    mean = [0.485, 0.456, 0.406] # mean of image tensor
    std = [0.229, 0.224, 0.225]  # standard devisation of image tensor   

    for images, labels in dataloader:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])

        # Denormalize image
        # 3, H, W, B
        tensor = images.clone().permute(1, 2, 3, 0)
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        
        # B, 3, H, W
        denormalized_images = torch.clamp(tensor, 0, 1).permute(3, 0, 1, 2)
        make_grid_denorm = make_grid(denormalized_images[:64], nrow=8)
        ax.imshow(make_grid_denorm.permute(1, 2, 0).clamp(0, 1))
        break








