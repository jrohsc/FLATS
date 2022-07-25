from collections import OrderedDict
import torch
import torch.nn as nn
from collections import defaultdict
from torchattacks import *
import time
import pickle
import random
import copy
from glob import glob
from tqdm.notebook import tqdm
from train_eval import *

# FedAvg (iid-setting)
def fedAvg(client_model_list, nk_n_list):
    new_params = OrderedDict()
    
    n = len(client_model_list)  # number of clients
    
    for client_model in client_model_list:
        local_param = client_model.state_dict()  # get current parameters of one client
        for k, v in local_param.items():
            new_params[k] = new_params.get(k, 0) + v / n
    
    return new_params

def federated_learning(atk, test_loader, clean_train_batch_ratio, num_attacks, white_model, num_selected, num_rounds, num_clients, num_local_epochs, global_model, client_model_list, client_loader_dict, device, nk_n_list):
    torch.cuda.empty_cache()
    
    id_loss_dict = defaultdict(list)
    id_acc_dict = defaultdict(list)
    update_count_dict = {}
    adv_train_dict = {}
    
    num_client_list = [i for i in range(num_clients)] # Total client id list
    
    #############################################
    # Method 2
#     attack_id_selected = random.sample(num_client_list, k=num_attacks) # Clients to be adversarially trained
#     print("Adversarial Training id selected: ", attack_id_selected)
#     for cli in attack_id_selected: 
#         adv_train_dict[cli] = 0
    #############################################    
    
    for _client in num_client_list:
        update_count_dict[_client] = 0
        # ***************************************
        # Method_1
        adv_train_dict[_client] = 0 
        # ***************************************

    
    local_w_list = [None for i in range(num_clients)]
    local_loss_list = [100 for i in range(num_clients)]
    
    glob_acc_list = []
    glob_loss_list = []
    robust_acc_list = []
    robust_loss_list = []
    
    criterion = nn.CrossEntropyLoss()
    
    # For each round
    start_time = time.time()
    for each_round in tqdm(range(num_rounds)):
        # Randomly selected client id "EACH ROUND"        
        # client_id_selected = num_client_list
        client_id_selected = random.sample(num_client_list, k=num_selected)   # Clients to be trained
        
        # ***************************************
        # Method_1
        attack_id_selected = random.sample(client_id_selected, k=num_attacks) # Clients to be adversarially trained (select within client to be trained)
        # ***************************************
        
        print("Selected client_id: ", client_id_selected)
        print("Adversarial Training id selected: ", attack_id_selected)
        
        # 1. Increase count for number of updates
        # 2. Increase count for adv_trained client id
        for _id in client_id_selected:
            update_count_dict[_id] += 1
            
            #############################################
#             # Method_2
#             if _id in attack_id_selected: 
#                 adv_train_dict[_id] += 1
            #############################################
        
        # ***************************************
        # Method_1
        for __id in attack_id_selected: 
            adv_train_dict[__id] += 1
        # ***************************************
        
        # For each client
        start_time = time.time()
        client_count = 1
        for client_id in tqdm(client_id_selected):
            print("")
            print(f"Updating [client_id]: {client_id}")
            print("")
            local_model = client_model_list[client_id]
            local_dataloader = client_loader_dict[client_id]
            
            learning_rate = 0.05
            optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
            
            # For each local epoch for each client
            for e in tqdm(range(num_local_epochs)):
                # Train local client models
                local_updated_model, new_local_loss, local_w, local_acc = local_train(clean_train_batch_ratio,
                                                                                      atk,
                                                                                      attack_id_selected, 
                                                                                      white_model,
                                                                                      client_id,
                                                                                      e, 
                                                                                      num_local_epochs, 
                                                                                      local_model, 
                                                                                      local_dataloader, 
                                                                                      device, 
                                                                                      criterion, 
                                                                                      optimizer)
                current_loss = local_loss_list[client_id]
                
                # Save loss & acc for each client id
                id_loss_dict[client_id].append(new_local_loss)
                id_acc_dict[client_id].append(local_acc)
                
                # Append new local_loss and local_weight
                local_loss_list[client_id] = new_local_loss
                local_w_list[client_id] = local_w 
                
                print(f"[rounds]: {each_round + 1}/{num_rounds} - [client_count]: {client_count}/{num_selected} - [local_epoch]: {e+1}/{num_local_epochs} - [local_loss]: {new_local_loss} - [local_acc]: {local_acc*100}%")
            
            client_count += 1
            
        # Federaed Averaging
        # new_glob_w = fedAvg(local_w_list, nk_n_list, client_id_selected)
        new_glob_w = fedAvg(client_model_list, nk_n_list)
        global_model.load_state_dict(new_glob_w)
        
        # Send new global model back to clients
        print("")
        print("Sending global model weight to local client models...")
        print("")
        for loc_model in client_model_list:
            loc_model.load_state_dict(new_glob_w)
        
        # Evaluate Global Model
        glob_acc, glob_loss = global_evaluate(global_model, test_loader, criterion, device)
        robust_acc, robust_loss = robust_evaluate(atk, white_model, global_model, test_loader, criterion, device)
        
        glob_acc_list.append(glob_acc*100)
        glob_loss_list.append(glob_loss)
        robust_acc_list.append(robust_acc*100)
        robust_loss_list.append(robust_loss)
        
        print("")
        print("*"*100)
        print("")
        print(f"[rounds]: {each_round + 1}/{num_rounds} - [global_loss]: {glob_loss} - [global_acc]: {glob_acc*100} %")
        print(f"[rounds]: {each_round + 1}/{num_rounds} - [robust_loss]: {robust_loss} - [robust_acc]: {robust_acc*100} %")
        print("")
        print("*"*100)
        print("")
    
    # Show number of updates for each client
    print("")
    for __id, count in update_count_dict.items():
        print("Client {} updated count: {}".format(__id, count))
    print("")
    
    # Adversarial Training Count
    print("")
    for _id, count in adv_train_dict.items():
        print("Client {} adversarial training count: {}".format(_id, count))
    print("")
    
    # Total time taken
    print("Time taken = {:.4f} minutes".format((time.time() - start_time) / 60))
    
    # Save Current Model
    pickle.dump(global_model, open('global_model.pkl', 'wb'))
    
    return global_model, glob_acc_list, robust_acc_list, glob_loss_list, robust_loss_list, id_loss_dict