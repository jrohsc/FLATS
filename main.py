from logging.config import valid_ident
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from data_preprocessing import preprocessing, dataloader, imshow, batch_imshow
from model import FaceRecog
from train_and_eval import train, evaluate, fit
from visualization import show_graph


if __name__ is '__main__':
    main_folder_path = 'pins_face_recognition (105_classes)'
    
    dataset, classes = preprocessing(main_folder_path=main_folder_path)
    train_loader, val_loader = dataloader(dataset=dataset)

    #################################################### Visualization ####################################################
    # # imshow
    # idx = 1
    # image, label = dataset[idx]
    # plt.imshow(image)
    # plt.title(dataset.classes[label])
    # plt.axis('off')

    # # batch_imshow
    # batch_imshow(train_loader)
    #######################################################################################################################
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FaceRecog(num_classes=len(classes)).to(device)

    # Model Summary
    # print(model.summary((3, 224, 224)))

    epoch = 20
    learning_rate = 1e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    result = fit(num_epochs=epoch,
                  model=model, 
                  train_loader=train_loader, 
                  valid_loader=val_loader, 
                  loss_func=criterion, 
                  device=device, 
                  optimizer=optimizer, 
                  dataset=dataset)
    
    # Visualization
    # show_graph(result)
    

