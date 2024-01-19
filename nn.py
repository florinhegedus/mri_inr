import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_channels: int=3, hidden_layer_size: int=256, num_hidden_layers=3, output_channels=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.layers = []

        # First layer
        self.layers.append(nn.Linear(input_channels, hidden_layer_size))
        self.layers.append(nn.BatchNorm1d(hidden_layer_size))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
            self.layers.append(nn.BatchNorm1d(hidden_layer_size))
            self.layers.append(nn.ReLU())

        # Last layer
        self.layers.append(nn.Linear(hidden_layer_size, output_channels))
        self.layers.append(nn.Sigmoid())

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out
