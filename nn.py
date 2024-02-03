import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from utils import logging


class NeuralNet(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=256, num_hidden_layers=18, output_channels=1):
        super().__init__()

        self.hidden_layer_size = hidden_layer_size

        self.layers = []

        # First layer
        self.initial_layer = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.BatchNorm1d(hidden_layer_size),
            nn.ReLU()
        )

        self.middle_layers = nn.ModuleList()
        for i in range(1, num_hidden_layers + 1):
            in_features = input_size + hidden_layer_size if i in [7, 13] else hidden_layer_size
            out_features = hidden_layer_size
            self.middle_layers.append(nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU()
            ))

        # Last layer
        self.final_layer = nn.Linear(hidden_layer_size, output_channels)

    def forward(self, x):
        skip_input = x
        # Initial layer
        out = self.initial_layer(x)

        # Middle layers with skip connections
        for i, layer in enumerate(self.middle_layers):
            if i == 6:  # Skip connection after 6th layer
                out = torch.cat((skip_input, out), dim=1)
            elif i == 12:  # Skip connection after 12th layer
                out = torch.cat((skip_input, out), dim=1)
            out = layer(out)

        # Final layer
        out = self.final_layer(out)
        return out
    
    def save_weights(self, path):
        logging.info(f"Saving model weights to {path}")
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        logging.info(f"Loading model weights from {path}")
        self.load_state_dict(torch.load(path))
    

class FourierMapping(ABC):
    @abstractmethod
    def map(self, P):
        """
        Applies a mapping/transformation to one or more 3D points.
        
        :param P: Input point(s) as a tensor. For a single point, the shape should be (3,).
                  For multiple points, the shape should be (N, 3), where N is the number of points.
        :return: Transformed features of the input point(s).
        """
        pass
    

class GaussianMapping(FourierMapping):
    def __init__(self, num_frequencies, scale, device):
        # Sample B from Gaussian distribution
        self.B = torch.randn(num_frequencies, 3, device=device) * scale

    def map(self, P):
        # Compute 2πBP (shape will be (..., num_features, 3))
        BP = (2 * torch.pi * P.float()) @ self.B.T

        # Concatenate cos and sin features
        fourier_features = torch.cat([torch.cos(BP), torch.sin(BP)], dim=-1)

        return fourier_features
    

class PosEncMapping(FourierMapping):
    def __init__(self, num_frequencies, scale):
        # Generate log-linear spaced frequencies
        self.frequencies = np.logspace(0, np.log10(scale), num_frequencies, base=np.e)
        self.num_frequencies = num_frequencies

    def map(self, P):
        """
        Applies positional encoding to one or more 3D points.

        :param P: Input point(s) as a tensor. For a single point, the shape should be (3,).
                  For multiple points, the shape should be (N, 3), where N is the number of points.
        :return: Encoded features of the input point(s). The shape of the output tensor will be (N, 2 * m * 3),
                 where each row corresponds to the encoded features of a point.
        """
        P = P.float()  # Ensure input is float
        encoded_features = []

        for freq in self.frequencies:
            # Calculate encoding for each frequency
            freq_encoding = 2 * np.pi * freq / self.num_frequencies
            cos_features = torch.cos(freq_encoding * P)
            sin_features = torch.sin(freq_encoding * P)
            encoded_features.append(cos_features)
            encoded_features.append(sin_features)

        # Concatenate all features
        encoded_features = torch.cat(encoded_features, dim=-1)

        return encoded_features


class FourierMappingFactory:
    @staticmethod
    def create(encoding, input_size=None, scale=None, device=None):
        if encoding == 'positional_encoding':
            if input_size is None:
                input_size = 120  # Default value if not provided
            return PosEncMapping(num_frequencies=input_size // 6, scale=scale if scale else 1000)
        elif encoding == 'gaussian':
            if input_size is None:
                input_size = 256  # Default value if not provided
            return GaussianMapping(num_frequencies=input_size // 2, scale=scale if scale else 10, device=device)
        else:
            raise ValueError("Unknown encoding type. Please choose 'pos_enc' or 'gaussian'.")
