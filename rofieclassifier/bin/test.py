# test.py
"""
Script Name: test.py
Description: This script performs inference using the Rofie model on test data.
Author: Salim Abboudi
Date: 2023-07-30
Version: 1.0
"""

import sys
import os
import cv2
import torch
import argparse
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import glob
from tqdm.auto import tqdm
from PIL import Image
from pprint import pprint
from torchsummary import summary
from typing import Tuple, List

# Add the 'rofieclassifier' directory to the Python path
rofieclassifier_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(rofieclassifier_path)

from model.model import Rofie
from utils.dataloader import Dataloader

def load_model(model_path: str) -> Tuple[Rofie, int]:
    """
    Load the Rofie model from the specified path and return it along with the image size.

    Args:
        model_path (str): Path to the trained model weights.

    Returns:
        Tuple[Rofie, int]: A tuple containing the loaded Rofie model and the image size.
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    img_size = checkpoint['imgsize']
    rofie = Rofie(img_size).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    rofie.eval()
    rofie.load_state_dict(checkpoint['model_state_dict'])
    return rofie, img_size


def perform_inference(rofie: Rofie, test_transform: transforms.Compose, image_paths: List[str], class_labels: List[str]) -> Tuple[int, List[Tuple[Image, int, int]]]:
    """
    Perform inference on the provided image paths and display the results.

    Args:
        rofie (Rofie): The loaded Rofie model.
        test_transform (transforms.Compose): Transformations to apply on test images.
        image_paths (List[str]): List of paths to test images.
        class_labels (List[str]): List of class labels.

    Returns:
        Tuple[int, List[Tuple[Image, int, int]]]: A tuple containing the accuracy counter and a list of image tuples with true and predicted labels.
    """
    acc_counter = 0
    images_with_labels = []
    for image_path in tqdm(image_paths, desc='Performing Inference', unit='image'):
        label = 0 if "field" in image_path else 1
        image = cv2.imread(image_path)
        if image is None:
            continue

        image0 = image.copy()
        image0 = cv2.resize(image0, (500, 500))

        image_tensor = test_transform(Image.fromarray(image)).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Make a prediction
        with torch.no_grad():
            rofie.eval()
            output = rofie(image_tensor)

            # round probabilities to get predicted class labels (0 or 1)
            _, pred_label = torch.max(output, 1)

        if label == pred_label:
            acc_counter += 1

        # Append the image with true and predicted labels to the list for plotting
        images_with_labels.append((image0, label, int(pred_label.item())))

        # Draw the true label on the image (in blue color)
        cv2.putText(image0, f"True Label: {class_labels[label]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 0, 0), 2)

        # Draw the predicted label on the image (in green color)
        cv2.putText(image0, f"Predicted Label: {class_labels[int(pred_label.item())]}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(f"Image with labels ", image0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return acc_counter, images_with_labels

def main() -> None:
    """
    Perform the main inference process using the trained model and test data.

    The function loads the trained model, performs inference on the test data, and displays the results.
    """
    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset.yaml',
                        help='path to the YAML configuration file for the test dataset Or Folder containing test image Or a single inference photo')
    parser.add_argument('--model_path', type=str, default='output/model_rofie.pth',
                        help='path to the trained model weights')
    args = vars(parser.parse_args())

    # Load the Rofie model
    
    rofie,img_size = load_model(args['model_path'])

    # Print Model Summary
    print("Model Summary:")
    summary(rofie, input_size=(3, img_size, img_size))

    # Testing transforms
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    class_labels = ['field', 'road']
    is_directory = True

    # Check if the user specified a single image path
    if args['data'].endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        is_directory = False
        image = cv2.imread(args['data'])
        image0 = image.copy()
        image0 = cv2.resize(image0, (500, 500))
        image_tensor = test_transform(Image.fromarray(image)).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

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
    if is_directory:
        if os.path.isdir(args['data']):
            image_paths = list(glob.iglob(args['data'] + '/**/*.*', recursive=True))
        elif not args['data'].endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            test_dir = Dataloader(args['data']).test_dir
            image_paths = list(glob.iglob(test_dir + '/**/*.*', recursive=True))

            if not any("field" in path or "road" in path for path in image_paths):
                raise ValueError("Image path must contain either 'field' or 'road' folder.")

        acc_counter, images_with_labels = perform_inference(rofie, test_transform, image_paths, class_labels)

        test_acc_avg = acc_counter / len(image_paths) * 100
        print(f"General test accuracy: {test_acc_avg:.3f} %")

        # Plot the first 10 images with true and predicted labels in subplots (2 rows, 5 columns)
        plt.figure(figsize=(12, 6))
        for i, (image, true_label, predicted_label) in enumerate(images_with_labels[:10], 1):
            plt.subplot(2, 5, i)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')

        # Add a global title to the figure
        plt.suptitle("Predicted Labels for Test Images")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust spacing for the global title
        plt.savefig('predicted_labels.png')
        plt.show()

if __name__ == "__main__":
    main()

