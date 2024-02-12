import logging
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn, optim
from mri import MRI
from utils import init_logging, evaluate, save_reconstruction_comparison, get_device


init_logging()


class GaussianModel(nn.Module):
    def __init__(self, mri_data, k_sigma, k_intensity, tau, max_intensity=None, device='cpu') -> None:
        super(GaussianModel, self).__init__()

        self.device = device
        self.mri_data = torch.tensor(mri_data, dtype=torch.float, device=self.device, requires_grad=False)  # Ensure MRI data is a tensor but not trainable
        self.max_intensity = max_intensity if max_intensity else self.mri_data.max()
        self.volume_shape = self.mri_data.shape

        # Calculate the scale factors to convert normalized coordinates back to volume indices
        self.scale_factors = torch.tensor(self.volume_shape, dtype=torch.float, device=self.device) - 1

        # Ensure that we're working with the grid coordinates correctly
        self.grid = torch.stack(torch.meshgrid([
            torch.linspace(0, 1, steps=mri_data.shape[0], device=self.device),
            torch.linspace(0, 1, steps=mri_data.shape[1], device=self.device),
            torch.linspace(0, 1, steps=mri_data.shape[2], device=self.device)
        ], indexing='ij'), dim=-1)  # Shape: [X, Y, Z, 3]

        centers, sigmas, intensities = self.setup_functions(k_sigma, k_intensity, tau)
        self.centers = nn.Parameter(centers.requires_grad_(True)).to(self.device)
        self.sigmas = nn.Parameter(sigmas.requires_grad_(True)).to(self.device)
        self.intensities = nn.Parameter(intensities.requires_grad_(True)).to(self.device)

    def setup_functions(self, k_sigma, k_intensity, tau):
        logging.info("Initializing Gaussians")
        mri_volume = self.mri_data.cpu().numpy()

        # Exclude empty spaces
        mri_volume = np.where(mri_volume > tau, mri_volume, 0)

        center_idxs, normalized_centers = self.get_gaussian_centers(mri_volume)

        logging.info("Calculating gaussian radiuses")
        sigmas = self.calculate_sigmas(normalized_centers)
        sigmas = sigmas * k_sigma

        logging.info("Intensities are calculated directly from the voxel values")
        intensities = torch.tensor([mri_volume[tuple(center)] for center in center_idxs], device=self.device)
        intensities = intensities * k_intensity

        # Normalize intensities
        intensities = intensities / self.max_intensity
        self.mri_data = self.mri_data / self.max_intensity

        assert len(normalized_centers) == len(sigmas) == len(intensities)

        logging.info(f"Initial number of gaussians: {len(normalized_centers)}")

        return normalized_centers, sigmas, intensities
    
    def get_gaussian_centers(self, mri_volume):
        logging.info("Compute gradient magnitude")
        grad_x, grad_y, grad_z = np.gradient(mri_volume)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

        logging.info("Filter out voxels with low gradients")
        # tau = grad_magnitude.mean()  # Example threshold
        filtered_grad_magnitude = np.where(grad_magnitude > 0.05, grad_magnitude, 0)

        # Find Gaussian Centers Based on Gradient Magnitude
        logging.info("Select points with gradient magnitude around the median as centers")
        median_grad = np.median(filtered_grad_magnitude[filtered_grad_magnitude > 0])
        close_to_median = np.abs(filtered_grad_magnitude - median_grad) < (0.01 * median_grad)
        center_idxs = np.argwhere(close_to_median)
        centers_tensor = torch.tensor(center_idxs, dtype=torch.long, device=self.device)

        # Extract the normalized coordinates for each center
        normalized_centers = self.grid[centers_tensor[:, 0], centers_tensor[:, 1], centers_tensor[:, 2]]

        return center_idxs, normalized_centers
    
    def calculate_sigmas(self, normalized_centers):
        normalized_centers_np = normalized_centers.cpu().numpy()  # Make sure it's on CPU and convert to NumPy

        # Create a k-d tree from the normalized centers
        tree = cKDTree(normalized_centers_np)

        # Query the k-d tree for the 4 closest points to each center (including itself)
        distances, indices = tree.query(normalized_centers_np, k=4)

        # Exclude the nearest distance (distance to itself) and take the next three
        closest_distances = distances[:, 1:4]  # Exclude the first column, which is zero

        # Calculate the mean of these distances for each center
        sigmas = np.mean(closest_distances, axis=1)

        # If you need to use sigmas in PyTorch, convert them back to a tensor
        sigmas_tensor = torch.tensor(sigmas, dtype=torch.float, device=normalized_centers.device)

        return sigmas_tensor
    
    def forward(self):
        volume = torch.zeros(self.volume_shape, dtype=torch.float, device=self.device)

        for i in range(len(self.centers)):
            center_norm = self.centers[i]  # Normalized center
            sigma = self.sigmas[i]
            intensity = self.intensities[i]

            # Convert normalized center to actual volume index space
            center = center_norm * self.scale_factors

            # Define cutoff boundary as 3 sigma, scaled to volume space
            cutoff = 3 * sigma * self.scale_factors

            # Calculate the min and max indices for slicing the grid based on the cutoff
            min_idx = torch.max(center - cutoff, torch.tensor([0, 0, 0], dtype=torch.float, device=self.device))
            max_idx = torch.min(center + cutoff, self.scale_factors)

            # Ensure indices are integers and within the volume bounds
            min_x, min_y, min_z = min_idx.int().tolist()
            max_x, max_y, max_z = (max_idx + 1).int().tolist()  # +1 to include the upper bound

            # Adjust max indices to ensure they do not exceed the volume shape
            max_x = min(max_x, self.volume_shape[0])
            max_y = min(max_y, self.volume_shape[1])
            max_z = min(max_z, self.volume_shape[2])

            # Use integer indices to slice the grid
            local_grid = self.grid[min_x:max_x, min_y:max_y, min_z:max_z, :]

            # Compute the squared distance from each point in the local grid to the Gaussian center
            distance_squared = torch.sum((local_grid - center_norm) ** 2, dim=-1)

            # Compute the Gaussian contribution using the squared distance
            contribution = intensity * torch.exp(-0.5 * distance_squared / (sigma ** 2))

            # Update the volume with the contribution of the current Gaussian within the neighborhood
            volume[min_x:max_x, min_y:max_y, min_z:max_z] += contribution

        return volume

    def loss(self, reconstructed_volume):
        # Implement the loss function. For example, use Mean Squared Error (MSE) between the reconstructed volume and the original MRI data
        mse_loss = nn.MSELoss()
        return mse_loss(reconstructed_volume, self.mri_data)


def main():
    lr_mri_path = '../../hcp1200/996782/T1w_acpc_dc_restore_brain_downsample_factor_2.nii.gz'
    lr_mri = MRI.from_nii_file(lr_mri_path)

    device = get_device()

    k_sigma = 0.4
    k_intensity = 0.4 
    tau = 0.05 # threshold for excluding empty spaces
    max_intensity = 1 # divide intensities by this value

    gm = GaussianModel(lr_mri.data, k_sigma, k_intensity, tau, max_intensity)
    
    # Separate parameters into groups for different learning rates
    center_params = [gm.centers]
    sigma_and_intensity_params = [gm.sigmas, gm.intensities]

    # Initialize the optimizer with separate parameter groups
    optimizer = optim.Adam([
        {'params': center_params, 'lr': 2e-4},  # Learning rate for centers
        {'params': sigma_and_intensity_params, 'lr': 0.0005}  # Constant learning rate for sigmas and intensities
    ], betas=(0.9, 0.999))

    # Define the number of epochs or iterations for optimization
    num_epochs = 50

    # Define an exponential decay for the learning rate of centers
    lambda1 = lambda epoch: (2e-6 / 2e-4) ** (epoch / num_epochs)
    lambda2 = lambda epoch: 1  # No change in learning rate

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Clear gradients
        reconstructed_volume = gm.forward()  # Perform forward pass
        loss = gm.loss(reconstructed_volume)  # Compute loss
        loss.backward()  # Backpropagate to compute gradients
        for name, param in gm.named_parameters():
            if param.grad is not None:
                print(f"Gradient for {name}: {param.grad.norm().item()}")
            else:
                print(f"No gradient for {name}")
        optimizer.step()  # Update parameters
        scheduler.step()

        logging.info(f"Epoch {epoch}, Loss: {loss.item()}")

    # Reconstruct volume
    with torch.no_grad():
        reconstructed_volume = gm.forward()
        reconstructed_volume = reconstructed_volume * gm.max_intensity
        reconstructed_volume = reconstructed_volume.numpy()
        pred_mri = MRI.from_data_and_voxel_size(reconstructed_volume, reconstructed_volume.shape)
    
    evaluate(pred_mri, lr_mri)
    save_reconstruction_comparison(lr_mri, pred_mri, lr_mri)
    return


if __name__ == '__main__':
    main()
