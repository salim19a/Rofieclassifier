import os
import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def save_model(epochs: int, model: torch.nn.Module, img_size: int, optimizer: torch.optim.Optimizer,
               loss: float, output_dir: str, name: str = "model_rofie.pth") -> None:
    """
    Save the trained PyTorch model to disk.

    Parameters:
        epochs (int): Number of training epochs completed.
        model (torch.nn.Module): The trained model to be saved.
        img_size (int): Input image size used during training.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        loss (float): The final loss value after training.
        output_dir (str): Directory path to save the model.
        name (str, optional): File name for the saved model. Defaults to "model_rofie.pth".
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_checkpoint = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'img_size': img_size
    }

    torch.save(model_checkpoint, os.path.join(output_dir, name))
