import matplotlib.pyplot as plt
import uuid
import datetime


def plot_metrics(unique_id, history, path="docs/loss_vs_accuracy_diagrams/"):
    """Plot training and validation loss and accuracy with a unique filename.

    Parameters:
    - history: History object from model training.
    - id: Unique identifier.
    - date: Current date.

    Returns:
    - None
    """
    # image wird nicht gespeichert.
    filename = path + f"{unique_id}.png"

    figure, axs = plt.subplots(2, 1)
    axs[0].set_title("Loss")
    axs[0].plot(history.history["loss"], label="train loss")
    axs[0].plot(history.history["val_loss"], label="test loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[1].set_title("Accuracy")
    axs[1].plot(history.history["accuracy"], label="train accuracy")
    axs[1].plot(history.history["val_accuracy"], label="test accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()


