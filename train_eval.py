import torch
import torch.nn as nn
from tqdm.notebook import tqdm

# Code Referance: https://www.kaggle.com/code/pezhmansamadi/facerecognition-torch-resnet34

def local_train(clean_train_batch_ratio, atk, attack_id_selected, white_model, client_id, e, local_epochs, local_model, train_loader, device, criterion, optimizer):
    torch.cuda.empty_cache()
    local_model.train(True)
    torch.set_grad_enabled(True)
    
    if client_id in attack_id_selected:
        # Create adversarial example
        print(atk)
        print("Adversarial Training on current device")
    
    total = 0
    correct = 0
    local_loss = 0.0
    local_acc = 0
    
    tq_batch = tqdm(train_loader, total=len(train_loader))
    for idx, (images, labels) in enumerate(tq_batch):
        images = images.to(device)
        labels = labels.to(device)
        
        ratio = int(len(train_loader) * clean_train_batch_ratio)
        
        # Attack if current id is the selected id
        if (client_id in attack_id_selected) and (idx >= ratio):
            # print("Half way")
            # Create adversarial example
            images = atk(images, labels)
            
        optimizer.zero_grad()
        outs = local_model(images)
        _, preds = torch.max(outs, 1)
        
        # outs = torch.exp(outs)
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()
        local_loss += loss.item()
        
        total += labels.size(0)
        correct += (preds == labels).sum()
        local_acc += float(correct) / total
        
        tq_batch.set_description(f'Local Epoch [{e + 1}/{local_epochs}]')
        tq_batch.set_postfix_str('Local loss = {:.4f} ; Local acc = {:.4f} '.format(loss.item(), float(correct) / total))
    
    # Average loss and acc of the training batch
    local_acc = local_acc / len(train_loader)
    local_loss = local_loss / len(train_loader)
    local_w = local_model.state_dict()

    return local_model, local_loss, local_w, local_acc

def global_evaluate(global_model, test_loader, criterion, device):
    torch.cuda.empty_cache()
    global_model.eval()
    
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

    return glob_acc, glob_loss

def robust_evaluate(atk, white_model, global_model, test_loader, criterion, device):
    torch.cuda.empty_cache()
    global_model.eval()
    
    total = 0
    robust_loss = 0
    robust_acc = 0
    correct = 0
    
    tq_batch = tqdm(test_loader, total=len(test_loader), leave=False)
    for images, labels in tq_batch:
        images = images.to(device)
        labels = labels.to(device)
        
        # Create adversarial example
        images = atk(images, labels)

        outputs = global_model(images)
        loss = criterion(outputs, labels)
        robust_loss += loss.item()
        
        _, preds = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (preds == labels).sum()
        
        robust_acc += float(correct) / total
        batch_acc = float(correct) / total
        
        tq_batch.set_postfix_str('Robust loss = {:.4f} ; Robust acc = {:.4f} '.format(loss.item(), batch_acc))
    
    # Average global loss and acc of the batch
    robust_loss = robust_loss / len(test_loader)
    robust_acc = robust_acc / len(test_loader)

    return robust_acc, robust_loss