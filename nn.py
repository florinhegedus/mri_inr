import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_channels: int=256, hidden_layer_size: int=256, num_hidden_layers=3, output_channels=1):
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
        out = fourier_feature_mapping(x)
        for layer in self.layers:
            out = layer(out)
        return out
    

def fourier_feature_mapping(P, L=128):
    """
    Applies Fourier feature mapping to one or more 3D points.

    This function maps each 3D point (or a batch of points) to a higher-dimensional
    space using Fourier features. It's useful for encoding spatial information in neural networks.

    :param P: Input point(s) as a tensor. For a single point, the shape should be (3,).
              For multiple points, the shape should be (N, 3), where N is the number of points.
    :param L: The number of frequencies to use in the Fourier feature mapping.
    :return: Fourier features of the input point(s). For a single point, the shape of the
             output tensor will be (2L,). For multiple points, the shape will be (N, 2L),
             where each row corresponds to the Fourier features of a point.
    """
    # Sample B from Gaussian distribution
    B = torch.randn(L, 3, device=P.device)

    # Compute 2πBP (shape will be (..., L, 3))
    BP = 2 * torch.pi * torch.matmul(P.float(), B.T)

    # Concatenate cos and sin features
    fourier_features = torch.cat([torch.cos(BP), torch.sin(BP)], dim=-1)

    return fourier_features
