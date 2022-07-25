import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

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

def dataloader(dataset, classes, train_batch_size, test_batch_size, num_clients):
    torch.manual_seed(42)
    
    # 70% to 30% train to test
    num_val = int(len(dataset) * 0.3)
    num_train = len(dataset) - num_val
    
    # 1. Train and Validation Data
    train_data, test_data = random_split(dataset, [num_train, num_val])
    
    # save train_data size (n)
    n = len(train_data)
    
    # Partition data for federated learning
    train_loader_list = []
    
    # nk / n list
    nk_n_list = []
    
    # Client ID: corresponding train_loader
    client_loader_dict = {}
    
    # Split train dataset
    total_train_size = len(train_data)
    examples_per_client = total_train_size // num_clients
    client_datasets = random_split(train_data, [min(i + examples_per_client, 
               total_train_size) - i for i in range(0, total_train_size, examples_per_client)])
    
    # Remove the left over (4)
    client_datasets = client_datasets[:num_clients]
    
    for idx, dataset in enumerate(client_datasets):
        print(f"Client_id {idx} data size: {len(dataset)}")
        
    print("")
    
    # Save dataloader for each
    for client_id, client_data in enumerate(client_datasets):
        # Save nk / n
        nk_n = len(client_data) / len(train_data)
        nk_n_list.append(nk_n)
        
        train_loader = DataLoader(client_data, batch_size=train_batch_size, shuffle=True, pin_memory=False, num_workers=3)
        train_loader_list.append(train_loader)
        client_loader_dict[client_id] = train_loader
    
    # Train and Validation Dataloader
    loader_idx = 0
    for key, _ in client_loader_dict.items():
        client_loader_dict[key] = train_loader_list[loader_idx]
        loader_idx += 1
    
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, pin_memory=False, num_workers=3)
    
    return train_loader_list, client_loader_dict, test_loader, test_data, nk_n_list