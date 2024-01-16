from utils import read_nii_file


def main():
    # Load the MRI data from the file
    file_path = '../../hcp1200/996782/T1w_acpc_dc_restore_brain.nii.gz'
    mri_volume, voxel_size = read_nii_file(file_path)
    print(mri_volume.shape)
    print(voxel_size)


if __name__ == "__main__":
    main()