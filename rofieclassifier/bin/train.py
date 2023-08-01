# train.py
"""
Script Name: train.py
Description: This script trains the Rofie model on the specified dataset.
Author: Salim Abboudi
Date: 2023-07-30
Version: 1.0
"""

import sys
import os
from typing import Tuple, List
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from pathlib import Path
from tqdm.auto import tqdm
from val import validate
from prettytable import PrettyTable
from torchsummary import summary
import torch.nn.init as init
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add the 'rofieclassifier' directory to the Python path
rofieclassifier_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(rofieclassifier_path)

from model.model import Rofie
from utils.dataloader import Dataloader
from utils.plots import save_plots
from model.save_model import save_model


def initialize_weights_xavier(model: nn.Module) -> None:
    """
    Initialize the weights of all layers using Xavier initialization.

    Args:
        model (nn.Module): The model whose weights need to be initialized.
    """
    for param in model.parameters():
        if len(param.shape) > 1:  # Ignore biases
            init.xavier_uniform_(param)

def train_epoch(
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Helper function to train the model for one epoch.

    Args:
        model (nn.Module): The model to be trained.
        trainloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        optimizer (optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss criterion for training.
        device (torch.device): Device to use for training.

    Returns:
        Tuple[float, float]: A tuple containing the average loss and accuracy for the epoch.
    """
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # forward pass
        outputs = model(image)

        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()

        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()

        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

def train_and_validate(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    device: torch.device,
    output_dir: str,
    save_period: int = None
) -> Tuple[nn.Module, List[float], List[float], List[float], List[float]]:
    """
    Main function to perform training and validation.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        valid_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        optimizer (optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss criterion for training.
        epochs (int): Number of training epochs.
        device (torch.device): Device to use for training and validation.
        output_dir (str): Output directory for plots, models, and results.
        save_period (int, optional): Number of epochs after which to save the model. Defaults to None.

    Returns:
        Tuple[nn.Module, List[float], List[float], List[float], List[float]]: A tuple containing the trained model,
        lists of training loss, validation loss, training accuracy, and validation accuracy.
    """
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    
    # Initialize a PrettyTable
    table = PrettyTable()
    table.field_names = ["Epoch", "Train Loss", "Train Acc", "Valid Loss", "Valid Acc"]
    
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        
        # Training
        train_epoch_loss, train_epoch_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader, criterion, device)
        
        # Scheduler l_r
        lr_scheduler.step(valid_epoch_loss)

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        
        # Add the results to the PrettyTable
        table.add_row([epoch+1, f"{train_epoch_loss:.3f}", f"{train_epoch_acc:.3f}",
                       f"{valid_epoch_loss:.3f}", f"{valid_epoch_acc:.3f}"])
        
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")

        print('-'*50)
        time.sleep(1)
        if save_period is not None:
            if (epoch % save_period) == 0 and epoch + 1 >= save_period and epoch + 1 != epochs:
                print("Saving Model...")
                save_model(epoch, model, args['imgsize'], optimizer, criterion, output_dir, f"rofie_epoch_{epoch}.pth")
    
    # Print the PrettyTable
    print("Training Metrics Summary:")
    print(table)
    model.eval()
    return model, train_loss, valid_loss, train_acc, valid_acc

if __name__ == "__main__":
     # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=20,
        help='number of epochs to train our network for')
    parser.add_argument('--imgsize', type=int, default=640,
        help='size of the input images')
    parser.add_argument('--data', type=str, default='dataset.yaml',
        help='path to the YAML configuration file')
    parser.add_argument('--batch', type=int, default=16,
        help='batch size for training and validation')
    parser.add_argument('--lr', type=float, default=1e-3,
        help='learning rate for training')
    parser.add_argument('--save_model', action='store_true',
        help='set this flag to save the trained model weights')
    parser.add_argument('--output_dir', type=str, default='output',
        help='Output directories for plots, models and results')
    parser.add_argument('--saveperiods', type=int, default=5,
        help='number of epochs after which to save the model')
    args = vars(parser.parse_args())
    output_dir = args['output_dir']
    # learning_parameters
    lr = args['lr']  # Use the parsed learning rate
    epochs = args['epochs']
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}\n")
    model = Rofie(args['imgsize']).to(device)

    # Call the initialize_weights_xavier function here to initialize the model's weights
    initialize_weights_xavier(model)

    summary(model, input_size=(3, args['imgsize'], args['imgsize']))
    # total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=1)

    # loss function
    criterion = nn.NLLLoss(reduction="sum")
    # Data loading
    data_config_file = args['data']
    data_loader_class = Dataloader(data_config_file)
    train_loader, valid_loader, _ = data_loader_class.get_loaders(batch_size=args['batch'], image_size=args['imgsize'])

    # Training and validation
    rofie, train_loss, valid_loss, train_acc, valid_acc = train_and_validate(model, train_loader, valid_loader,
                                                                             optimizer, criterion, epochs, device, args['output_dir'], args['saveperiods'])

    # Save the trained model weights if save_model is True
    if args['save_model']:
        save_model(epochs, rofie, args['imgsize'], optimizer, criterion, output_dir)

    # Save the loss and accuracy plots if save_model is True
    save_plots(train_acc, valid_acc, train_loss, valid_loss, output=output_dir, save=True)

    print('TRAINING COMPLETE')
