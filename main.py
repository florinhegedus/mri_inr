import nibabel as nib
import matplotlib.pyplot as plt
from utils import display_brain_slices, read_nii_file


def main():
    # Load the MRI data from the file
    file_path = '../../hcp1200/996782/T1w_acpc_dc_restore_brain.nii.gz'
    display_brain_slices(file_path)
    
    mri_data = read_nii_file(file_path)
    print(mri_data.shape)


if __name__ == "__main__":
    main()