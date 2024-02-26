import logging
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn, optim
from mri import MRI
from utils import init_logging, evaluate, save_reconstruction_comparison, get_device


init_logging()


class GaussianModel(nn.Module):
    def __init__(self, mri_data, k_sigma, k_intensity, tau, N_max=5000, max_intensity=None, device='cpu') -> None:
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

        self.centers = nn.Parameter(torch.zeros(N_max, 3).requires_grad_(True)).to(self.device)
        self.sigmas = nn.Parameter(torch.zeros(N_max).requires_grad_(True)).to(self.device)
        self.intensities = nn.Parameter(torch.zeros(N_max).requires_grad_(True)).to(self.device)

        centers, sigmas, intensities = self.setup_functions(k_sigma, k_intensity, tau)

        self.centers.data[:centers.size(0), :] = centers
        self.sigmas.data[:sigmas.size(0)] = sigmas
        self.intensities.data[:intensities.size(0)] = intensities

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
        close_to_median = np.abs(filtered_grad_magnitude - median_grad) < (0.05 * median_grad)
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
        num_active_gaussians = self.sigmas[self.sigmas > 0].size(0)

        for i in range(num_active_gaussians):
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
    
    def clone_gaussian(self, index, centers, sigmas, intensities):
        center = centers[index].clone().detach()
        sigma = sigmas[index].clone().detach()
        intensity = intensities[index].clone().detach() / 2

        grad_direction = self.centers.grad[index] / self.centers.grad[index].norm()
        new_center = center + grad_direction * 0.01

        # Correctly wrap the updated tensors with nn.Parameter
        centers = torch.cat((centers, new_center.unsqueeze(0)), dim=0)
        sigmas = torch.cat((sigmas, sigma.unsqueeze(0)), dim=0)
        intensities = torch.cat((intensities, intensity.unsqueeze(0)), dim=0)

        return centers, sigmas, intensities

    def split_gaussian(self, index, centers, sigmas, intensities, phi=1.6):
        center = centers[index]
        sigma = sigmas[index] / phi  # Reduce sigma by a factor phi
        intensity = intensities[index] / 2  # Halve the intensity for both splits
        
        # Direction for splitting can be based on some heuristic or random variation
        split_direction = torch.randn_like(center)
        split_direction /= split_direction.norm()
        
        # Initialize positions of the new Gaussians by slightly offsetting them from the original
        new_center1 = center + split_direction * sigma
        new_center2 = center - split_direction * sigma
        
        # Remove the original Gaussian and add the new ones
        centers = torch.cat((centers[:index], centers[index+1:], new_center1.unsqueeze(0), new_center2.unsqueeze(0)), dim=0)
        sigmas = torch.cat((sigmas[:index], sigmas[index+1:], sigma.unsqueeze(0), sigma.unsqueeze(0)), dim=0)
        intensities = torch.cat((intensities[:index], intensities[index+1:], intensity.unsqueeze(0), intensity.unsqueeze(0)), dim=0)

        return centers, sigmas, intensities
    
    def prune_gaussians(self, epsilon_alpha, centers, sigmas, intensities):
        # Identify Gaussians to prune based on intensity threshold
        mask = intensities > epsilon_alpha
        
        # Remove identified Gaussians
        centers = centers[mask]
        sigmas = sigmas[mask]
        intensities = intensities[mask]

        return centers, sigmas, intensities

    def adaptive_density_control(self, tau_pos=0.0001, epsilon_alpha=0.01, phi=1.5):
        """
        Adjusts the density of Gaussians based on reconstruction needs.

        Args:
        - tau_pos (float): Threshold for positional gradient magnitude to trigger densification.
        - epsilon_alpha (float): Threshold below which Gaussians are considered for pruning.
        - phi (float): Division factor for the sigma of split Gaussians.
        """
        logging.info("Running adaptive density control")
        # Ensure gradients are computed
        if self.centers.grad is None:
            return

        # Calculate the norm of the positional gradients
        grad_norms = self.centers.grad.norm(dim=1)

        # Get active gaussians
        num_active_gaussians = self.sigmas[self.sigmas > 0].size(0)

        # Get the values for centers, sigmas and intensities
        centers = self.centers.data[:num_active_gaussians]
        sigmas = self.sigmas.data[:num_active_gaussians]
        intensities = self.intensities.data[:num_active_gaussians]

        splitted = 0
        cloned = 0
        idx = 0

        while idx < num_active_gaussians:
            # Determine action based on gradient magnitude and Gaussian size
            if grad_norms[idx] > tau_pos:                # For large Gaussians or those in over-reconstructed areas, consider splitting
                if sigmas[idx] > (sigmas.mean() * phi):
                    splitted += 1
                    centers, sigmas, intensities = self.split_gaussian(idx, centers, sigmas, intensities, phi)
                    continue
                elif self.sigmas[idx] < (self.sigmas.mean() / phi):
                    # For small Gaussians or under-reconstructed areas, consider cloning
                    cloned += 1
                    centers, sigmas, intensities = self.clone_gaussian(idx, centers, sigmas, intensities)
            idx += 1

        no_before_pruning = sigmas.size(0)
        centers, sigmas, intensities = self.prune_gaussians(epsilon_alpha, centers, sigmas, intensities)

        logging.info(f"Number of gaussians splitted={splitted}")
        logging.info(f"Number of gaussians cloned={cloned}")
        logging.info(f"Number of gaussians pruned={no_before_pruning - sigmas.size(0)}")
        logging.info(f"Total number of gaussians: {len(centers)}")

        self.centers.data[:centers.size(0), :] = centers
        self.sigmas.data[:sigmas.size(0)] = sigmas
        self.intensities.data[:intensities.size(0)] = intensities

    def loss(self, reconstructed_volume):
        # Implement the loss function. For example, use Mean Squared Error (MSE) between the reconstructed volume and the original MRI data
        mse_loss = nn.MSELoss()
        return mse_loss(reconstructed_volume, self.mri_data)
    

def reconstruct(gm, lr_mri, epoch):
    # Reconstruct volume
    with torch.no_grad():
        reconstructed_volume = gm.forward()
        reconstructed_volume = reconstructed_volume * gm.max_intensity
        reconstructed_volume = reconstructed_volume.numpy()
        pred_mri = MRI.from_data_and_voxel_size(reconstructed_volume, reconstructed_volume.shape)
    
    evaluate(pred_mri, lr_mri)
    save_reconstruction_comparison(lr_mri, pred_mri, lr_mri, epoch)


def main():
    lr_mri_path = '../../hcp1200/996782/T1w_acpc_dc_restore_brain_downsample_factor_4.nii.gz'
    lr_mri = MRI.from_nii_file(lr_mri_path)

    device = get_device()

    k_sigma = 0.4
    k_intensity = 0.4 
    tau = 0.05 # threshold for excluding empty spaces
    N_max = 25000 # max number of gaussians
    max_intensity = 1 # divide intensities by this value, None for dividing with max(intensities)
    adaptive_density_control = True
    densify_frequency = 100

    gm = GaussianModel(lr_mri.data, k_sigma, k_intensity, tau, N_max, max_intensity, device='cpu')
    
    # Separate parameters into groups for different learning rates
    center_params = [gm.centers]
    sigma_and_intensity_params = [gm.sigmas, gm.intensities]

    # Initialize the optimizer with separate parameter groups
    optimizer = optim.Adam([
        {'params': center_params, 'lr': 2e-4},  # Learning rate for centers
        {'params': sigma_and_intensity_params, 'lr': 0.0005}  # Constant learning rate for sigmas and intensities
    ], betas=(0.9, 0.999))

    # Define the number of epochs or iterations for optimization
    num_epochs = 5000

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

        if epoch % densify_frequency == densify_frequency - 1 and adaptive_density_control:
            reconstruct(gm, lr_mri, epoch)
            gm.adaptive_density_control()

        logging.info(f"Epoch {epoch}, Loss: {loss.item()}")

    reconstruct(gm, lr_mri, num_epochs)

    return


if __name__ == '__main__':
    main()
