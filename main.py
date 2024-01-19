from mri import MRI


def main():
    # Load the MRI data from the file
    file_path = '../../hcp1200/996782/T1w_acpc_dc_restore_brain.nii.gz'
    mri = MRI(file_path)
    mri.downsample(4)


if __name__ == "__main__":
    main()