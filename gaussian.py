from mri import MRI
from utils import init_logging


init_logging()


class GaussianModel:
    def __init__(self, mri_data) -> None:
        self.mri_data = mri_data

    def get_initial_points(self):
        high_intensity = self.mri_data[self.mri_data > 0.0]
        return high_intensity


def main():
    lr_mri_path = '../../hcp1200/996782/T1w_acpc_dc_restore_brain_downsample_factor_4.nii.gz'
    lr_mri = MRI.from_nii_file(lr_mri_path)
    gm = GaussianModel(lr_mri.data)
    gm.get_initial_points()
    return


if __name__ == '__main__':
    main()