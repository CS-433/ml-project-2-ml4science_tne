import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, n_classes, layers=(64, 32, 16)):
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
        return self.model(x)

class CNN(nn.Module):
    def __init__(self, input_channels, n_classes, length, channels=(8, 8, 16),
                 layers=(64, 32), paddings=(1, 1, 1), convkernels=(3, 3, 3),
                 maxpoolkernels=(2, 2, 2)):
        super().__init__()

        self.convolution = nn.Sequential(*[
            nn.Conv1d(input_channels, channels[0], convkernels[0], padding=paddings[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(maxpoolkernels[0]),
            nn.BatchNorm1d(channels[0]),
            *[m for c1, c2, k, mk, p in zip(channels, channels[1:], convkernels[1:], maxpoolkernels[1:], paddings[1:]) for m in [
                nn.Conv1d(c1, c2, k, padding=p),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(mk),
                nn.BatchNorm1d(c2)
            ]]
        ])

        # Forward a dummy tensor across the convolutions to flexibly find the output size
        output_size = self.convolution(torch.tensor(np.zeros((1, 1, length))).float()).flatten().shape[0]

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
        x = self.convolution(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)
