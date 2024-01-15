import nibabel as nib
import matplotlib.pyplot as plt
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)


def display_brain_slices(file_path):
    # Load the MRI data from the file
    file_path = '../../hcp1200/996782/T1w_acpc_dc_restore_brain.nii.gz'
    mri_data = read_nii_file(file_path)

    # Select an axial slice - middle of the brain
    sagittal_slice = mri_data[mri_data.shape[0] // 2, :, :]
    coronal_slice = mri_data[:, mri_data.shape[1] // 2, :]
    axial_slice = mri_data[:, :, mri_data.shape[2] // 2]

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

    logging.info("Saving MRI image with brain slices")
    fig.savefig("images/brain.png")

def read_nii_file(file_path):
    logging.info(f"Reading data from {file_path}")
    mri_image = nib.load(file_path)

    # Get the data as a numpy array
    mri_data = mri_image.get_fdata()
    return mri_data