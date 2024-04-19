import matplotlib.pyplot as plt

def show_result(x, y, pred_y):
    plt.cla()
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.title("Ground truth", fontsize = 18)
    for i in range(len(x)):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1, 2, 2)
    plt.title("Predict result", fontsize = 18)
    for i in range(len(x)):
        if pred_y[i] < 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.savefig("result.png")

def show_loss(losses, name = "loss"):
    plt.cla()
    plt.clf()
    plt.plot(losses, label = name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss", fontsize = 18)
    plt.legend()
    plt.savefig("loss.png")
