import logging
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.processing
from utils import init_logging


init_logging()


class MRI:
    def __init__(self, nii_file_path: str):
        self.nii_file_path = nii_file_path
        self.data, self.voxel_size = self.__read_nii_file(nii_file_path)

    def __read_nii_file(self, file_path):
        logging.info(f"Reading data from {file_path}")
        mri_file = nib.load(file_path)

        # Get the data as a numpy array
        data = mri_file.get_fdata()
        voxel_size = mri_file.header.get_zooms()
        return data, voxel_size
    
    def display_relevant_brain_slices(self):
        logging.info(f"Displaying relevant brain slices")
        # Select an axial slice - middle of the brain
        sagittal_slice = self.data[self.data.shape[0] // 2, :, :]
        coronal_slice = self.data[:, self.data.shape[1] // 2, :]
        axial_slice = self.data[:, :, self.data.shape[2] // 2]

        # Create a figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        # Display each slice
        axes[0].imshow(axial_slice.T, cmap="gray", origin="lower")
        axes[0].set_title("Axial Slice")
        axes[0].axis("off")

        axes[1].imshow(sagittal_slice.T, cmap="gray", origin="lower")
        axes[1].set_title("Sagittal Slice")
        axes[1].axis("off")

        axes[2].imshow(coronal_slice.T, cmap="gray", origin="lower")
        axes[2].set_title("Coronal Slice")
        axes[2].axis("off")

        plt.show()

    def downsample(self, factor: int):
        logging.info(f"Downsampling MRI by a factor of {factor}")
        hr = nibabel.load(self.nii_file_path)
        voxel_size = [i * factor for i in self.voxel_size]
        lr = nibabel.processing.resample_to_output(hr, voxel_size)
        output_path = self.nii_file_path[:-7] + f"_downsample_factor_{factor}" + ".nii.gz"
        logging.info(f"Saving downsampled MRI to {output_path}")
        nibabel.save(lr, output_path)
    
    def __str__(self) -> str:
        return f"MRI(volume={self.data.shape}, voxel_size={self.voxel_size}"
