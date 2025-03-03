import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, n_classes, layers=(16, 8)):
        """Create an MLP model.

        Args:
            input_size (int): number of features per datapoint.
            n_classes (int): number of output classes.
            layers (tuple, optional): tuple with the number of neurons per layer. Defaults to (16, 8).
        """
        super().__init__()

        # Dynamically creates sequential layers
        self.model = nn.Sequential(*[
            nn.Linear(input_size, layers[0]),
            nn.ReLU(inplace=True),
            *[m for l1, l2 in zip(layers, layers[1:]) for m in [
                nn.Linear(l1, l2),
                nn.ReLU(inplace=True)
            ]],
            nn.Linear(layers[-1], n_classes)
        ])


    def forward(self, x):
        """Forward pass through the model.

        Args:
            x (Tensor): input tensor.

        Returns:
            Tensor: model's logits
        """
        return self.model(x)

class CNN1D(nn.Module):
    def __init__(self, input_channels, n_classes, length, channels=(8, 8, 16),
                 layers=(64, 32), paddings=(1, 1, 1), convkernels=(3, 3, 3), strides=(1, 1, 1),
                 maxpoolkernels=(2, 2, 2)):
        """Create a 1D CNN model.

        Args:
            input_channels (int): number of channels in input
            n_classes (int): number of output classes.
            length (int): length of the input timeserie.
            channels (tuple, optional): number of channels per convolutional layer.
            Defaults to (8, 8, 16).
            layers (tuple, optional): number of linear layer in the MLP. Defaults to (64, 32).
            paddings (tuple, optional): size of the padding per convolutional layer.
            Defaults to (1, 1, 1).
            convkernels (tuple, optional): size of the convolution kernel (1D) per convolutional layer.
            Defaults to (3, 3, 3).
            strides (tuple, optional): size of the stride per convolutional layer.
            Defaults to (1, 1, 1).
            maxpoolkernels (tuple, optional): size of the maxpool kernel per convolutional layer.
            Defaults to (2, 2, 2).
        """
        super().__init__()

        self.convolution = nn.Sequential(*[
            nn.Conv1d(input_channels, channels[0], convkernels[0], stride=strides[0], padding=paddings[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(maxpoolkernels[0]),
            nn.BatchNorm1d(channels[0]),
            *[m for c1, c2, k, mk, s, p in zip(channels, channels[1:], convkernels[1:], maxpoolkernels[1:], strides[1:], paddings[1:]) for m in [
                nn.Conv1d(c1, c2, k, stride=s, padding=p),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(mk),
                nn.BatchNorm1d(c2)
            ]]
        ])

        # Forward a dummy tensor across the convolutions to flexibly find the output size
        output_size = self.convolution(torch.tensor(np.zeros((1, input_channels, length))).float()).flatten().shape[0]

        self.fc = nn.Sequential(
            nn.Linear(output_size, layers[0]),
            nn.ReLU(inplace=True),
            *[m for l1, l2 in zip(layers, layers[1:]) for m in [
                nn.Linear(l1, l2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ]],
            nn.Linear(layers[-1], n_classes)
        )

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x (Tensor): input tensor.

        Returns:
            Tensor: model's logits
        """
        x = self.convolution(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

class CNN2D(nn.Module):
    def __init__(self, input_channels, n_classes, height, width, channels=(8, 8, 16),
                 layers=(64, 32), paddings=(1, 1, 1), convkernels=(3, 3, 3), strides=(1, 1, 1),
                 maxpoolkernels=(2, 2, 2)):
        """Create a 2D CNN model.

        Args:
            input_channels (int): number of channels in input
            n_classes (int): number of output classes.
            length (int): length of the input timeserie.
            channels (tuple, optional): number of channels per convolutional layer.
            Defaults to (8, 8, 16).
            layers (tuple, optional): number of linear layer in the MLP. Defaults to (64, 32).
            paddings (tuple, optional): size of the padding per convolutional layer.
            Defaults to (1, 1, 1).
            convkernels (tuple, optional): size of the convolution kernel (2D) per convolutional layer.
            If given a single number per layer, results in a square kernel. Defaults to (3, 3, 3).
            strides (tuple, optional): size of the stride per convolutional layer.
            Defaults to (1, 1, 1).
            maxpoolkernels (tuple, optional): size of the maxpool kernel per convolutional layer.
            Defaults to (2, 2, 2).
        """
        super().__init__()
        assert len(channels) == len(maxpoolkernels) == len(strides) == len(paddings) == len(convkernels)

        self.convolution = nn.Sequential(*[
            nn.Conv2d(input_channels, channels[0], convkernels[0], stride=strides[0], padding=paddings[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(maxpoolkernels[0]),
            nn.BatchNorm2d(channels[0]),
            *[m for c1, c2, k, mk, s, p in zip(channels, channels[1:], convkernels[1:], maxpoolkernels[1:], strides[1:], paddings[1:]) for m in [
                nn.Conv2d(c1, c2, k, stride=s, padding=p),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(mk),
                nn.BatchNorm2d(c2)
            ]]
        ])

        # Forward a dummy tensor across the convolutions to flexibly find the output size
        output_size = self.convolution(torch.tensor(np.zeros((1, input_channels, height, width))).float()).flatten().shape[0]

        self.fc = nn.Sequential(
            nn.Linear(output_size, layers[0]),
            nn.ReLU(inplace=True),
            *[m for l1, l2 in zip(layers, layers[1:]) for m in [
                nn.Linear(l1, l2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ]],
            nn.Linear(layers[-1], n_classes)
        )

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x (Tensor): input tensor.

        Returns:
            Tensor: model's logits
        """
        x = self.convolution(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)
