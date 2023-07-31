"""
Validation script 
"""
import sys
import os
# Add the 'rofieclassifier' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

from tqdm.auto import tqdm

def validate(model, testloader, criterion, device):
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
        
        # forward pass
        outputs = model(image)
        
        # calculate the loss
        loss = criterion(outputs, labels)
        test_running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        test_running_correct += (preds == labels).sum().item()
        
    
    # loss and accuracy for the complete epoch
    epoch_loss = test_running_loss / counter
    epoch_acc = 100. * (test_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc