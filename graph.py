import matplotlib.pyplot as plt
def loss_graph(loss):
    plt.plot(loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


def display_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.legend(['train', 'test'], loc='upper left')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')

    ax2.plot(history.history['accuracy'])
    ax2.plot(history.history['val_accuracy'])
    ax2.legend(['train', 'test'], loc='upper left')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')

    plt.show()