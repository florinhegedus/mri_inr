from data import get_mri_dataloader, get_test_dataloader
from mri import MRI
from nn import NeuralNet
from utils import get_device


def main():
    # Load the MRI data from the file
    file_path = '../../hcp1200/996782/T1w_acpc_dc_restore_brain_downsample_factor_8.nii.gz'
    mri = MRI(file_path)

    device = get_device()

    get_test_dataloader(mri, 1, 2048, device)


if __name__ == "__main__":
    main()