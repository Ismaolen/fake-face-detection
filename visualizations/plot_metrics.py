import matplotlib.pyplot as plt
import uuid
import datetime


def plot_metrics(unique_id, history, path="docs/loss_vs_accuracy_diagrams/"):
    """
    Plots and saves the training and validation loss and accuracy.

    This function takes the training history object of a Keras model and plots the training and validation
    loss, as well as the accuracy, on separate subplots. The resulting plot is saved to a specified path with
    a filename derived from a unique identifier.

    Parameters
    ----------
    unique_id : str
        A unique identifier for the filename under which the plot will be saved.
    history : History
        The History object obtained from the fit method of a Keras model. Contains loss and accuracy metrics.
    path : str, optional
        The path to the directory where the plot image will be saved. Defaults to "docs/loss_vs_accuracy_diagrams/".

    Returns
    -------
    None
    """
    # Construct the filename using the provided unique identifier
    filename = path + f"{unique_id}.png"

    # Create subplots for loss and accuracy
    figure, axs = plt.subplots(2, 1)  # Two subplots in one column
    axs[0].set_title("Loss")  # Title for the first subplot
    # Plot training and validation loss
    axs[0].plot(history.history["loss"], label="train loss")  # Training loss
    axs[0].plot(history.history["val_loss"], label="test loss")  # Validation loss
    axs[0].set_xlabel("Epochs")  # X-axis label
    axs[0].set_ylabel("Loss")  # Y-axis label
    axs[0].legend()  # Show legend

    axs[1].set_title("Accuracy")  # Title for the second subplot
    # Plot training and validation accuracy
    axs[1].plot(history.history["accuracy"], label="train accuracy")  # Training accuracy
    axs[1].plot(history.history["val_accuracy"], label="test accuracy")  # Validation accuracy
    axs[1].set_xlabel("Epochs")  # X-axis label
    axs[1].set_ylabel("Accuracy")  # Y-axis label
    axs[1].legend()  # Show legend

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(filename)  # Save the plot to the specified file
    plt.show()  # Display the plot
    plt.close()  # Close the plot to free up resources
