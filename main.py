from data import get_mri_dataloader
from mri import MRI
from nn import NeuralNet
from utils import get_device


def main():
    # Load the MRI data from the file
    file_path = '../../hcp1200/996782/T1w_acpc_dc_restore_brain_downsample_factor_8.nii.gz'
    mri = MRI(file_path)

    device = get_device()
    dataloader = get_mri_dataloader(mri, batch_size=8, device=device)

    net = NeuralNet()
    net.to(device)
    
    coords, gt_intensity_values = next(iter(dataloader))
    output = net(coords)


if __name__ == "__main__":
    main()