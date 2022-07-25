from preprocessing import *
from federated_learning import *
from train_eval import *
from model import *
from torchattacks import *
from visualization import *
from robust_eval import *
import argparse

if __name__ is "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_folder_path')
    parser.add_argument('--num_clients', default=5)
    parser.add_argument('--train_batch_size', default=64)
    parser.add_argument('--test_batch_size', default=64)

    parser.add_argument('--num_selected', default=5)
    parser.add_argument('--num_attack', default=1)
    parser.add_argument('--num_rounds', default=10)
    parser.add_argument('--num_local_epochs', default=5)
    parser.add_argument('--clean_train_batch_ratio', default=5)
    parser.add_argument('--atk', default=FFGSM(white_model, eps=8/255, alpha=10/255))

    args = parser.parse_args()

    # Hyperparameters
    main_folder_path = args.main_folder_path
    num_clients = args.num_clients
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_sie

    num_selected = args.num_selected
    num_attack = args.num_attack
    num_rounds = args.num_rounds
    num_local_epochs = args.num_local_epochs
    clean_train_batch_ratio = args.clean_train_batch_ratio

    print("*"*100)
    print("main_folder_path: ", main_folder_path)
    print("num_clients: ", num_clients)
    print("train_batch_size: ", train_batch_size)
    print("test_batch_size: ", test_batch_size)
    print("")

    print("num_selected: ", num_selected)
    print("num_attack: ", num_attack)
    print("num_rounds: ", num_rounds)
    print("num_local_epochs: ", num_local_epochs)
    print("clean_train_batch_ratio: ", clean_train_batch_ratio)
    print("*"*100)
    print("")

    # *********************************************************************************************************
    # Data Preprocessing
    # *********************************************************************************************************
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset, classes = preprocessing(main_folder_path=main_folder_path)
    train_loader_list, client_loader_dict, test_loader, _, nk_n_list = dataloader(dataset, classes, train_batch_size, test_batch_size, num_clients)
    num_classes = len(classes)

    for idx, dt in client_loader_dict.items():
        print(f'Client {idx} dataloader size: {len(dt)}')

    print("")
    print("Test dataloader size: ", len(test_loader))

    # *********************************************************************************************************
    # Robust Federated Learning
    # *********************************************************************************************************
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model = black_box_model(num_classes=num_classes).to(device)
    client_model_list = [black_box_model(num_classes=num_classes).to(device) for _ in range(num_clients)]

    # Model used to create adversarial examples
    # White-box models: AlexNet
    white_model = models.alexnet(pretrained=True).to(device)
    
    atk_list = [ # torchattack list
        FGSM(white_model, eps=8/255),
    #     BIM(white_model, eps=8/255, alpha=100, steps=100),
    #     RFGSM(white_model, eps=8/255, alpha=2/255, steps=100),
    #     CW(white_model, c=1, lr=0.01, steps=100, kappa=0),
    #     PGD(white_model, eps=8/255, alpha=2/225, steps=10, random_start=True),
    #     PGDL2(white_model, eps=1, alpha=0.2, steps=100),
    #     EOTPGD(white_model, eps=8/255, alpha=2/255, steps=100, eot_iter=2),
        FFGSM(white_model, eps=8/255, alpha=10/255), # *
    #     TPGD(white_model, eps=8/255, alpha=2/255, steps=100),
    #     MIFGSM(white_model, eps=8/255),
    #     VANILA(white_model),
    #     FAB(white_model, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=False),
    #     FAB(white_model, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=True),
        Square(white_model, eps=8/255, n_queries=2000, n_restarts=1, loss='ce'), # *
    #     OnePixel(white_model, pixels=5, inf_batch=50),
    #     DeepFool(white_model, steps=10),
    #     DIFGSM(white_model, eps=8/255, alpha=2/255, steps=10, diversity_prob=0.5, resize_rate=0.9)
    #     DIFGSM(white_model, eps=8/255)
    ]

    # Attack received from parsing
    atk = args.atk
    print("atk: ", atk)

    # atk = FFGSM(white_model, eps=8/255, alpha=10/255) # Adversarial Training Method
    start_time = time.time()
    global_model, glob_acc_list, robust_acc_list, glob_loss_list, robust_loss_list, id_loss_dict = federated_learning(atk, 
                                                                                                                    test_loader,
                                                                                                                    clean_train_batch_ratio,
                                                                                                                    num_attack,
                                                                                                                    white_model,
                                                                                                                    num_selected,
                                                                                                                    num_rounds, 
                                                                                                                    num_clients, 
                                                                                                                    num_local_epochs, 
                                                                                                                    global_model, 
                                                                                                                    client_model_list, 
                                                                                                                    client_loader_dict, 
                                                                                                                    device,
                                                                                                                    nk_n_list)

    print("Time taken = {:.4f} minutes".format((time.time() - start_time) / 60))

    # *********************************************************************************************************
    # Visualization
    # *********************************************************************************************************
    acc_visual(num_rounds, glob_acc_list, robust_acc_list)
    loss_visual(num_rounds, glob_loss_list, robust_loss_list)

    # *********************************************************************************************************
    # Robust Eval
    # *********************************************************************************************************
    standard_eval(test_loader, global_model, device)
    robust_eval(atk_list, device, global_model, test_loader)

