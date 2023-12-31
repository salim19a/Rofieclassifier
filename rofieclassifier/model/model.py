import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

params = {
    "initial_filters": 8,
    "num_fc1": 100,
    "dropout_rate": 0.25,
    "num_classes": 2
}

class Rofie(nn.Module):
    def __init__(self, imgsize: int) -> None:
        """
        Rofie neural network constructor.

        Parameters:
            imgsize (int): The size of the input image (assumed to be square).

        Returns:
            None
        """
        super(Rofie, self).__init__()

        Cin, Hin, Win = (3, imgsize, imgsize)
        init_f = params["initial_filters"]
        num_fc1 = params["num_fc1"]
        num_classes = params["num_classes"]
        self.dropout_rate = params["dropout_rate"]

        # Convolution Layers
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        h, w = find_conv2d_out_shape(Hin, Win, self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2 * init_f, kernel_size=3)
        h, w = find_conv2d_out_shape(h, w, self.conv2)
        self.conv3 = nn.Conv2d(2 * init_f, 4 * init_f, kernel_size=3)
        h, w = find_conv2d_out_shape(h, w, self.conv3)
        self.conv4 = nn.Conv2d(4 * init_f, 8 * init_f, kernel_size=3)
        h, w = find_conv2d_out_shape(h, w, self.conv4)

        # Compute the flatten size
        self.num_flatten = h * w * 8 * init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Rofie neural network.

        Parameters:
            X (torch.Tensor): The input tensor of shape (batch_size, 3, imgsize, imgsize).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_classes).
        """
        # Convolution & Pool Layers
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)

        X = X.view(-1, self.num_flatten)

        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)

def find_conv2d_out_shape(hin: int, win: int, conv: nn.Conv2d, pool: int = 2) -> tuple:
    """
    Compute the output shape of a 2D convolutional layer.

    Parameters:
        hin (int): Input height.
        win (int): Input width.
        conv (nn.Conv2d): The convolutional layer.
        pool (int, optional): The pooling size. Defaults to 2.

    Returns:
        tuple: Output height and width as integers.
    """
    # Get conv arguments
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation

    hout = np.floor((hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    wout = np.floor((win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    if pool:
        hout /= pool
        wout /= pool
    return int(hout), int(wout)

