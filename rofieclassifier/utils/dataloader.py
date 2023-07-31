"""
Datalaoders for training and validation 
"""
import numpy as np 
import os
import yaml
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,ConcatDataset
from utils.general import  check_yaml, check_img_size

class Dataloader:

    def __init__(self, data_config_file):
        try:
            # Attempt to open and load data from the YAML file
            with open(check_yaml(data_config_file), errors='ignore') as f:
                data_config = yaml.safe_load(f)  # data dict

        except Exception as e:  
            # If any exception occurs during the loading process, raise a new Exception with a specific message
            raise Exception("error  loading the yaml file...") from e

        self.train_dir = data_config['train']
        self.valid_dir = data_config['val']
        self.test_dir = data_config['test']


    def get_loaders(self, batch_size, image_size):
        
        self.image_size=check_img_size(image_size)
        # Training transforms
        train_transform = transforms.Compose([
                                transforms.Resize((self.image_size,self.image_size)),
                                
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                transforms.ToTensor()])
        valid_transform = transforms.Compose([
                                transforms.Resize((self.image_size,self.image_size)),
                                
                                transforms.ToTensor()])

        
         # Add label mapping to convert 'field' to 0 and 'road' to 1
    
    

        # Create dataset for training

        train_dataset = ImageFolder(root=self.train_dir, transform=train_transform)
       

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
         # Create dataset for validation

        valid_dataset = ImageFolder(root=self.valid_dir, transform=valid_transform)

        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        # Create dataset for test

        test_dataset = ImageFolder(root=self.test_dir, transform=valid_transform)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        return train_loader,valid_loader,test_loader

    

# Example usage:

if __name__ == "__main__":
    data_config_file = "dataset.yaml"  # Specify the path of your data.yaml file
    data_loader_class = Dataloader(data_config_file)

    # Get training data loader with batch size 5 and image size (256, 256)
    train_loader,valid_loader,_ = data_loader_class.get_loaders(batch_size=4,image_size=128)

    print(train_loader.dataset[15][0])
    for images, labels in train_loader:
        # images and labels have shape (batch_size, 3, image_size, image_size) and (batch_size,) respectively
        print("Shape of the images tensor:", images.shape)
        print("Shape of the labels tensor:", labels)
        break  # Only check the first batch




    
     