import torch
from tqdm.notebook import tqdm
import torch.nn as nn
from torchvision import models

def standard_eval(test_loader, global_model, device):
    # atk = FGSM(model, eps=8/255)
    torch.cuda.empty_cache()
    global_model.eval()
    criterion = nn.CrossEntropyLoss()

    total = 0
    glob_loss = 0
    glob_acc = 0
    correct = 0

    tq_batch = tqdm(test_loader, total=len(test_loader), leave=False)
    for images, labels in tq_batch:
        images = images.to(device)
        labels = labels.to(device)

        outputs = global_model(images)
        loss = criterion(outputs, labels)
        glob_loss += loss.item()

        _, preds = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (preds == labels).sum()

        glob_acc += float(correct) / total
        batch_acc = float(correct) / total

        tq_batch.set_postfix_str('Global loss = {:.4f} ; Global acc = {:.4f} '.format(loss.item(), batch_acc))

    # Average global loss and acc of the batch
    glob_loss = glob_loss / len(test_loader)
    glob_acc = glob_acc / len(test_loader)

    print("Global loss: ", glob_loss)
    print("Global accuracy: {} %".format(glob_acc*100))


def robust_eval(atk_list, device, global_model, test_loader):
    torch.cuda.empty_cache()
    global_model.eval()
    criterion = nn.CrossEntropyLoss()

    total = 0
    glob_loss = 0
    glob_acc = 0
    correct = 0

    # tq_batch = tqdm(test_loader, total=len(test_loader))

    for atk in tqdm(atk_list):
        print("")
        print("*"*100)
        print(atk)
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Attack the model
            images = atk(images, labels)

            outputs = global_model(images)
            loss = criterion(outputs, labels)
            glob_loss += loss.item()

            _, preds = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (preds == labels).sum()

            glob_acc += float(correct) / total
            batch_acc = float(correct) / total

            # tq_batch.set_postfix_str('Robust loss = {:.4f} ; Robust acc = {:.4f} '.format(loss.item(), batch_acc))

        # Average global loss and acc of the batch
        glob_loss = glob_loss / len(test_loader)
        glob_acc = glob_acc / len(test_loader)
        
        print("Robust loss: ", glob_loss)
        print("Robust accuracy: {}%".format(glob_acc*100))
        print("*"*100)
        print("")