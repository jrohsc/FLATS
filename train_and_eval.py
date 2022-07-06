import torch
import torch.nn as nn
from tqdm import tqdm

# Code Referance: https://www.kaggle.com/code/pezhmansamadi/facerecognition-torch-resnet34

def train(epoch, n_epochs, model, train_loader, loss_func, device, criterion, optimizer, dataset):
    model.train(True)
    torch.set_grad_enabled(True)
    
    epoch_loss = 0.0
    epochs_acc = 0
    
    tq_batch = tqdm(train_loader, total=len(train_loader))
    for images, labels in tq_batch:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outs = model(images)
        _, preds = torch.max(outs, 1)
        
        loss = loss_func(outs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epochs_acc += torch.sum(preds == labels).item()
        
        tq_batch.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
        tq_batch.set_postfix_str('loss = {:.4f}'.format(loss.item()))

            
    epoch_loss = epoch_loss / len(train_loader)
    epochs_acc = epochs_acc / len(dataset)

    return epoch_loss, epochs_acc

def evaluate(model, val_loader, loss_func, device, dataset):

    model.train(False)

    epoch_loss = 0
    epochs_acc = 0
    tq_batch = tqdm(val_loader, total=len(val_loader), leave=False)
    for images, labels in tq_batch:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        loss = loss_func(outputs, labels)

        epoch_loss += loss.item()
        epochs_acc += torch.sum(preds == labels).item()
        tq_batch.set_description(f'Evaluate Model')
        
    epoch_loss = epoch_loss / len(val_loader)
    epochs_acc = epochs_acc / len(dataset)

    return epoch_loss, epochs_acc

def fit(num_epochs, model, train_loader, val_loader, loss_func, device, optimizer, dataset):
    
    history = []
    val_loss_ref = float('inf') # float type infinity
    
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        loss, acc = train(epoch, num_epochs, model, train_loader, loss_func, device, optimizer, dataset)
        
        torch.cuda.empty_cache()
        val_loss, val_acc = evaluate(model, val_loader, loss_func, device, dataset)
        
        # Save all the results
        # For visualization in the future
        history.append({'loss': loss, 
                        'acc': acc, 
                        'val_loss': val_loss, 
                        'val_acc': val_acc})

        statement = "[loss]={:.4f} - [acc]={:.4f} - [val_loss]={:.4f} - [val_acc]={:.4f}".format(loss, acc, val_loss, val_acc,)
        print(statement)
        
        ####### Checkpoint
        if val_loss < val_loss_ref:
            val_loss_ref = val_loss
            model_path = './Face_Recognition_checkpoint.pth'
            torch.save(model.state_dict(), model_path)
            print(f"[INFO] Saving model dict, Epoch={epoch + 1}")

    return history