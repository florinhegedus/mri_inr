import logging
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)


def display_brain_slices(mri_data):
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
    mri_file = nib.load(file_path)

    # Get the data as a numpy array
    mri_volume = mri_file.get_fdata()
    voxel_dimensions = mri_file.header.get_zooms()
    return mri_volume, voxel_dimensions


def extract_planes(mri_data):
    coronal = mri_data[:, :, mri_data.shape[2] // 2]
    axial = mri_data[:, mri_data.shape[1] // 2, :]
    sagittal = mri_data[mri_data.shape[0] // 2, :, :]
    return coronal, axial, sagittal


def register_to_fixed(fixed_image, moving_image):
    # Convert the images to SimpleITK image types if they are not already
    fixed_image_sitk = sitk.Cast(sitk.GetImageFromArray(fixed_image), sitk.sitkFloat32) if not isinstance(fixed_image, sitk.Image) else fixed_image
    moving_image_sitk = sitk.Cast(sitk.GetImageFromArray(moving_image), sitk.sitkFloat32) if not isinstance(moving_image, sitk.Image) else moving_image

    # Initialize the registration method
    registration_method = sitk.ImageRegistrationMethod()

    # Set the similarity metric (how similar the images are)
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    # Set the optimizer (how the algorithm searches for the solution)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Use the geometric center of the images to initialize the transformation
    initial_transform = sitk.CenteredTransformInitializer(fixed_image_sitk, moving_image_sitk, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Set the interpolator (how the algorithm interpolates values in the image)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Execute the registration
    final_transform = registration_method.Execute(fixed_image_sitk, sitk.Cast(moving_image_sitk, sitk.sitkFloat32))
    
    return final_transform
