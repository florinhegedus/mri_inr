import torch
from torch.utils.data import Dataset, DataLoader
import logging


class MRIIntensityDataset(Dataset):
    def __init__(self, voxel_coords, intensities):
        """
        Initialize the dataset with voxel coordinates and corresponding intensity values.

        This dataset holds voxel coordinates from an MRI scan and their associated 
        intensity values. It's designed to facilitate the processing and analysis of MRI data.

        :param voxel_coords: A tensor containing voxel coordinates (shape: [N, 3])
                             where N is the number of voxels and each row represents the
                             (x, y, z) coordinates of a voxel.
        :param intensities: A tensor of intensity values (shape: [N])
                            corresponding to each voxel.
        """
        self.voxel_coords = voxel_coords
        self.intensities = intensities

    def __len__(self):
        """
        Returns the number of items in the dataset.
        """
        return len(self.intensities)

    def __getitem__(self, idx):
        """
        Retrieves the voxel coordinates and intensity value for the voxel at the specified index.

        :param idx: Index of the voxel
        :return: A tuple (voxel_coords, intensity) where voxel_coords is the 3D coordinates
                 of the voxel and intensity is its corresponding intensity value.
        """
        return self.voxel_coords[idx], self.intensities[idx]
    

def get_voxel_grid(mri, upsample_factor=1):
    '''
    Generates a voxel grid for the MRI data.

    This function creates a 3D grid of voxel coordinates based on the dimensions
    of the MRI dataset. It returns a 2D tensor where each row contains the (x, y, z)
    coordinates of a voxel in the MRI volume.

    Returns:
        torch.Tensor: A 2D tensor of shape (num_voxels, 3), where num_voxels is the
                      total number of voxels in the MRI volume and each row represents
                      the (x, y, z) coordinates of a voxel.
    '''
    # Get voxel grid
    logging.info(f"Creating voxel grid: originial voxel grid resolution x{upsample_factor}")
    # Define the dimensions of the MRI
    x_dim, y_dim, z_dim = [ax * upsample_factor for ax in mri.data.shape]

    # Create range vectors for each dimension
    x_range = torch.arange(x_dim)
    y_range = torch.arange(y_dim)
    z_range = torch.arange(z_dim)

    # Create a mesh grid
    x_grid, y_grid, z_grid = torch.meshgrid(x_range, y_range, z_range, indexing='ij')

    # Combine and reshape the grids to create a list of coordinates
    voxel_grid = torch.stack((x_grid, y_grid, z_grid), dim=-1).reshape(-1, 3)

    return voxel_grid


def normalize_coordinates(grid):
    """
    Normalize the coordinates of the grid to be between -1 and 1.

    :param grid: The coordinate grid (tensor of shape (N, 3))
    :return: Normalized grid (tensor of shape (N, 3))
    """
    # Find the maximum dimension sizes
    max_values = grid.max(0)[0]

    # Normalize the grid
    normalized_grid = (grid - max_values / 2) / (max_values / 2)
    normalized_grid = normalized_grid.float()
    return normalized_grid


def get_intensity_values(mri):
    intensity_values = torch.tensor(mri.data.flatten(), dtype=torch.float)
    return intensity_values


def get_train_dataloader(mri, batch_size, device):
    # Get voxel grid
    logging.info(f"Create voxel grid from MRI")
    voxel_grid = get_voxel_grid(mri).to(device)
    normalized_grid = normalize_coordinates(voxel_grid)

    # Get intensity values and move to the same device
    logging.info(f"Get intensity values from MRI")
    intensity_values = get_intensity_values(mri).to(device)

    # Create the dataset and dataloader
    logging.info(f"Creating the dataloader")
    dataset = MRIIntensityDataset(normalized_grid, intensity_values) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def get_test_dataloader(mri, upsample_factor, batch_size, device):
    voxel_grid = get_voxel_grid(mri, upsample_factor).to(device)
    normalized_grid = normalize_coordinates(voxel_grid)

    dataloader = DataLoader(normalized_grid, batch_size=batch_size, shuffle=False)

    return dataloader
