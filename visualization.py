import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import torch
from torchvision.utils import make_grid

# Code Referance: https://www.kaggle.com/code/pezhmansamadi/facerecognition-torch-resnet34

def show_graph(history):
    accuracy = [res['acc'] for res in history]
    losses = [res['loss'] for res in history]
    val_accuracy = [res['val_acc'] for res in history]
    val_losses = [res['val_loss'] for res in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
    ax1.plot(losses, '-o', label = 'Loss')
    ax1.plot(val_losses, '-o', label = 'Validation Loss')
    ax1.legend()

    ax2.plot(100 * np.array(accuracy), '-o', label = 'Accuracy')
    ax2.plot(100 * np.array(val_accuracy), '-o', label = 'Validation Accuracy')
    ax2.legend();
    
    fig.show()

def denormalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

def show_result(val_loader, device, model, classes):
    idx = 16
    for imgs, lbls in val_loader:
        imgs = imgs.to(device)
        outs = model(imgs)
        _, preds = torch.max(outs, dim = 1)
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        denorm_images = denormalize(imgs.cpu())
        ax.imshow(make_grid(denorm_images[:idx], nrow=8).permute(1, 2, 0).clamp(0,1))
        for p, lbl in zip(preds[:idx], lbls[:idx]):
            if lbl == p.cpu():
                print(colored(classes[p], 'green'))
            else:
                print(colored(classes[p], 'red'), colored(classes[lbl], 'blue'))
        break