import matplotlib.pyplot as plt
import os
from typing import List

def save_plots(train_acc: List[float], valid_acc: List[float], train_loss: List[float], valid_loss: List[float], 
               save: bool = False, output: str = "output") -> None:
    """
    Function to save the loss and accuracy plots to disk using Matplotlib.

    Parameters:
        train_acc (List[float]): List of training accuracies for each epoch.
        valid_acc (List[float]): List of validation accuracies for each epoch.
        train_loss (List[float]): List of training losses for each epoch.
        valid_loss (List[float]): List of validation losses for each epoch.
        save (bool, optional): If True, save the figures to disk. Default is False.
        output (str, optional): Output directory for saving the plots. Defaults to "output".
    """
    # Number of epochs (assuming lists are of equal length)
    epochs = len(train_acc)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot loss history on the first subplot
    axs[0].plot(range(1, epochs + 1), train_loss, label='Train Loss')
    axs[0].plot(range(1, epochs + 1), valid_loss, label='Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss History')
    axs[0].legend()

    # Plot metric history (e.g., accuracy) on the second subplot
    axs[1].plot(range(1, epochs + 1), train_acc, label='Train Accuracy')
    axs[1].plot(range(1, epochs + 1), valid_acc, label='Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Metric History')
    axs[1].legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()

    # Save the plot to disk if save is True
    if save:
        if not os.path.exists(output):
            os.makedirs(output)
        fig.savefig(f'{output}/plots.png')




