import logging
import numpy as np
from scipy.spatial import cKDTree
from mri import MRI
from utils import init_logging


init_logging()


class GaussianModel:
    def __init__(self, mri_data) -> None:
        self.mri_data = mri_data
        self.setup_functions()

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
        close_to_median = np.abs(filtered_grad_magnitude - median_grad) < (0.1 * median_grad)
        gaussian_centers = np.argwhere(close_to_median)

        logging.info("Calculating gaussian radiuses")
        radii = self.calculate_gaussian_radii_kdtree(gaussian_centers)

        logging.info("Intensities are calculated directly from the voxel values")
        intensities = np.array([mri_volume[tuple(center)] for center in gaussian_centers])

        assert len(gaussian_centers) == len(intensities) == len(radii)

        return gaussian_centers, intensities, radii
    
    def calculate_gaussian_radii_kdtree(self, gaussian_centers):
        kd_tree = cKDTree(gaussian_centers)

        # Define the search radius within which other centers are considered 'close'
        search_radius = 5  # Adjust this value based on your specific requirements and scale

        # Query the k-d tree for points within the search radius for each center
        # This returns a list of points for each center that are within the search radius
        counts = kd_tree.query_ball_tree(kd_tree, r=search_radius)

        # Calculate the radius inversely proportional to the number of close centers + 1 to avoid division by zero
        radii = np.array([1 / len(counts[i]) for i in range(len(gaussian_centers))])

        return radii


def main():
    lr_mri_path = '../../hcp1200/996782/T1w_acpc_dc_restore_brain_downsample_factor_2.nii.gz'
    lr_mri = MRI.from_nii_file(lr_mri_path)
    gm = GaussianModel(lr_mri.data)

    return


if __name__ == '__main__':
    main()