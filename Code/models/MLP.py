import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, n_classes, layers=(256, 128, 256)):
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