# val.py
"""
Script Name: val.py
Description: This script performs validation using the Rofie model on validation data.
Author: Salim Abboudi
Date: 2023-07-30
Version: 1.0
"""
import sys
import os
import torch
from tqdm.auto import tqdm

def validate(model: torch.nn.Module, testloader: torch.utils.data.DataLoader, criterion: torch.nn.Module,
             device: torch.device) -> tuple:
    """
    Perform validation on the given model using the test data.

    Parameters:
        model (torch.nn.Module): The trained model to be validated.
        testloader (torch.utils.data.DataLoader): DataLoader containing the test data.
        criterion (torch.nn.Module): Loss function used for evaluation.
        device (torch.device): The device (CPU or GPU) to use for computation.

    Returns:
        tuple: A tuple containing epoch_loss (float) and epoch_accuracy (float) for the complete epoch.
    """
    model.eval()
    print('Validation')
    test_running_loss = 0.0
    test_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(image)

            # Calculate the loss
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()

            # Calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            test_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch
    epoch_loss = test_running_loss / counter
    epoch_acc = 100. * (test_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc
