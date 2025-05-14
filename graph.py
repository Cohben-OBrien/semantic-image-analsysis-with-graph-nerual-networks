import matplotlib.pyplot as plt
def loss_graph(loss):
    plt.plot(loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


