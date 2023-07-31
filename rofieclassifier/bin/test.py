import cv2
import numpy as np
import torch
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import sys
import os
import glob
from tqdm.auto import tqdm
from PIL import Image
from pprint import pprint
from torchsummary import summary

# Add the 'rofieclassifier' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model import Rofie
from utils.dataloader import Dataloader

def main():
    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset.yaml',
     help='path to the YAML configuration file for the test dataset Or Folder containing test image Or a single inference photo')
    parser.add_argument('--model_path', type=str, default='output/model_rofie.pth',
                        help='path to the trained model weights')
    args = vars(parser.parse_args())
    
    checkpoint = torch.load(args['model_path'], map_location='cpu')

    Metadata = {key: value for key, value in checkpoint.items() if key in ['loss', 'epoch', 'imgsize']}

     

    print("Formatted Metadata : ")
    pprint(Metadata)
    
    imgsize=checkpoint['imgsize']

    # Device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    rofie = Rofie(imgsize).to(device)
    rofie.eval()
    rofie.load_state_dict(checkpoint['model_state_dict'])
    
    # Print Model
    print("Model Summary : ")

    summary(Rofie(imgsize),input_size=(3,imgsize,imgsize))
    # Testing transforms
    test_transform = transforms.Compose([
        transforms.Resize((imgsize, imgsize)),
        transforms.ToTensor(),
    ])
   
    
    # Lists to keep track of losses and accuracies

    class_labels = ['field', 'road']
    isdirectory=True
    # Check if the user specified a single image path
    if args['data'].endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        isdirectory=False
        image = cv2.imread(args['data'])
        image0 = image.copy()
        image0 = cv2.resize(image0, (500, 500))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = test_transform(Image.fromarray(image)).to(device)
        
        # Make a prediction
        with torch.no_grad():
            output = rofie(image_tensor.unsqueeze(0))
            _, pred = torch.max(output.data, 1)
            pred_label = class_labels[pred.item()]

        # Plot the image with the predicted label
        cv2.imshow(f"Predicted Label: {pred_label}", image0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Directory Test 
    if isdirectory:
        if os.path.isdir(args['data']):
            image_paths = list(glob.iglob(args['data'] + '/**/*.*', recursive=True))
        elif not args['data'].endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            test_dir=Dataloader(args['data']).test_dir
            image_paths = list(glob.iglob(test_dir + '/**/*.*', recursive=True))
            
        
            if not any("field" in path or "road" in path for path in image_paths):
                raise ValueError("Image path must contain either 'field' or 'road' folder.")
        
        acc_counter=0
        counter=0
        for imagepath in image_paths:
            counter+=1
            # Class Label
            print(imagepath)
            label = 0 if "field" in imagepath else 1  
            image = cv2.imread(imagepath)
            if image is None :
                pass
            else:

                image0 = image.copy()
                image0 = cv2.resize(image0, (500, 500))
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_tensor = test_transform(Image.fromarray(image)).to(device)
            
                # Make a prediction
                with torch.no_grad():
                    rofie.eval()
                    output = rofie(image_tensor)
                    
                    # round probabilities to get predicted class labels (0 or 1)
                    _, pred_label = torch.max(output, 1)
                    print(output)
                    
                if label==pred_label: acc_counter+=1
                # Plot the image with the predicted label
                cv2.imshow(f"True: {class_labels[label]} &  Predicted: {class_labels[int(pred_label.item())]}", image0)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                # Calculate general accuracy and loss for the test
            
        test_acc_avg = acc_counter /counter * 100

        print(f" General test accuracy: {test_acc_avg:.3f} %")
    

if __name__ == "__main__":

    main()


