import logging
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from mri import MRI
from utils import init_logging


init_logging()


class GaussianModel:
    def __init__(self, mri_data) -> None:
        self.mri_data = mri_data
        self.centers, self.sigmas, self.intensities = self.setup_functions()
        # centers = nn.Parameter(torch.tensor(self.centers, dtype=torch.float, requires_grad=True))

    def setup_functions(self):
        logging.info("Initializing Gaussians")
        mri_volume = self.mri_data

        logging.info("Compute gradient magnitude")
        grad_x, grad_y, grad_z = np.gradient(mri_volume)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

        logging.info("Filter out low gradients")
        tau = grad_magnitude.mean()
        filtered_grad_magnitude = np.where(grad_magnitude > tau, grad_magnitude, 0)

        # Find Gaussian Centers Based on Gradient Magnitude
        logging.info("Select points with gradient magnitude around the median as centers")
        median_grad = np.median(filtered_grad_magnitude[filtered_grad_magnitude > 0])
        close_to_median = np.abs(filtered_grad_magnitude - median_grad) < (0.5 * median_grad)
        centers = np.argwhere(close_to_median)

        logging.info("Calculating gaussian radiuses")
        sigmas = self.calculate_gaussian_radii_kdtree(centers, search_radius=5)

        logging.info("Intensities are calculated directly from the voxel values")
        intensities = np.array([mri_volume[tuple(center)] for center in centers])

        assert len(centers) == len(sigmas) == len(intensities)

        logging.info(f"Initial number of gaussians: {len(centers)}")

        return centers, sigmas, intensities
    
    def calculate_gaussian_radii_kdtree(self, gaussian_centers, search_radius):
        kd_tree = cKDTree(gaussian_centers)

        # Query the k-d tree for points within the search radius for each center
        # This returns a list of points for each center that are within the search radius
        counts = kd_tree.query_ball_tree(kd_tree, r=search_radius)

        # Calculate the radius inversely proportional to the number of close centers + 1 to avoid division by zero
        radii = np.array([1 / len(counts[i]) for i in range(len(gaussian_centers))])

        return radii

    def gaussian_contribution(self, x, center, sigma, intensity):
        """
        Calculate the contribution of a single Gaussian at a given position x.
        
        Parameters:
        - x: The position in the volume where the contribution is being calculated.
        - center: The center coordinates of the Gaussian (µi = (xi, yi, zi)).
        - sigma: The standard deviation of the Gaussian, reflecting an isotropic covariance matrix.
        - intensity: The intensity (Ii) of the Gaussian.
        
        Returns:
        - The contribution of the Gaussian to the volume at position x.
        """
        # Calculate the squared Euclidean distance between x and the Gaussian center
        distance_squared = np.sum((x - center) ** 2)
        # Calculate the contribution using the Gaussian function
        return intensity * np.exp(-0.5 * distance_squared / sigma ** 2)

    def discretize_gaussians(self, d_factor=3):
        """
        Discretize a set of 3D Gaussians into a volumetric image with optimization.
        Limit calculations to within a confidence interval around each Gaussian center.
        
        Parameters:
        - volume_shape: The shape of the volumetric image (x, y, z).
        - centers: An array of center coordinates for the Gaussians.
        - sigmas: An array of standard deviations for the Gaussians.
        - intensities: An array of intensities for the Gaussians.
        - d_factor: Multiplier to determine the extent of calculation around each center.
        
        Returns:
        - A volumetric image represented as a NumPy array, with the optimized contribution of all Gaussians.
        """
        volume_shape = self.mri_data.shape
        volume = np.zeros(volume_shape)
        for i in range(len(self.centers)):
            sigma = self.sigmas[i]
            center = self.centers[i]
            intensity = self.intensities[i]
            
            # Define the extent of calculation for the Gaussian based on d_factor
            d = d_factor * sigma
            lower_bounds = np.maximum(0, np.floor(center - d).astype(int))
            upper_bounds = np.minimum(volume_shape, np.ceil(center + d).astype(int))
            
            # Iterate only within the defined extent for the current Gaussian
            for x in np.ndindex(tuple(upper_bounds - lower_bounds)):
                x_global = lower_bounds + x  # Convert local coordinates to global coordinates
                contribution = self.gaussian_contribution(x_global, center, sigma, intensity)
                volume[tuple(x_global)] += contribution

        return volume
    
    def loss(self):
        
        return


def main():
    lr_mri_path = '../../hcp1200/996782/T1w_acpc_dc_restore_brain_downsample_factor_8.nii.gz'
    lr_mri = MRI.from_nii_file(lr_mri_path)
    gm = GaussianModel(lr_mri.data)
    reconstructed_volume = gm.discretize_gaussians()
    pred_mri = MRI.from_data_and_voxel_size(reconstructed_volume, (lr_mri.data.shape))
    return


if __name__ == '__main__':
    main()