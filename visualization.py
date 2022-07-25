import matplotlib.pyplot as plt

def acc_visual(num_rounds, glob_acc_list, robust_acc_list):
    num_round_list = list(range(1, num_rounds+1))

    ymin = 0
    ymax = 100

    plt.figure(figsize=(7, 7))
    plt.grid(axis='y')
    plt.gca().set_ylim([ymin, ymax])
    plt.title("Clean/Robust Accuracy of the Global Model")
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")

    plt.plot(num_round_list, glob_acc_list, '-o', label='Clean Gobal Acc.')
    plt.plot(num_round_list, robust_acc_list, '-o', label='Robust Global Acc.')
    plt.legend()

    plt.savefig('acc_graph.png')

def loss_visual(num_rounds, glob_loss_list, robust_loss_list):
    num_round_list = list(range(1, num_rounds+1))

    num_round_list = list(range(1, num_rounds+1))

    ymax = 5
    ymin = 0

    plt.figure(figsize=(7, 7))
    plt.grid(axis='y')

    plt.gca().set_ylim([ymin, ymax])
    plt.title("Clean / Robust Loss of the Global Model")
    plt.xlabel("Round")
    plt.ylabel("Loss")

    plt.plot(num_round_list, glob_loss_list, '-o', label='Clean Global Loss')
    plt.plot(num_round_list, robust_loss_list, '-o', label='Robust Global Loss')
    plt.legend()

    plt.savefig('loss_graph.png')