import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def init_logging():
    blue_color_code = '\033[94m'  # Light blue color code
    reset_code = '\033[0m'  # Reset color code

    logging.basicConfig(
        level=logging.INFO,
        format=f'{blue_color_code}%(asctime)s | %(message)s{reset_code}',
        datefmt='%H:%M:%S'
    )


def set_seed():
    torch.manual_seed(0)


def get_device(verbose: bool=True) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("--"*30)
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('\tAllocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('\tReserved: ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        else:
            print('Using CPU')
        print("--"*30)

    return device


def read_config():
    # Path to your YAML file
    config_file_path = 'config.yaml'

    # Read the YAML file
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def save_reconstruction_comparison(lr_mri, pred_mri, gt_mri, file_path="images/comparison.png"):
    logging.info("Saving comparison to images/comparison.png")
    lr_slice = lr_mri.data[lr_mri.data.shape[0] // 2, :, :]
    pred_slice = pred_mri.data[pred_mri.data.shape[0] // 2, :, :]
    gt_slice = gt_mri.data[gt_mri.data.shape[0] // 2, :, :]
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Display each slice
    axes[0].imshow(lr_slice.T, cmap="gray", origin="lower")
    axes[0].set_title("LR Input")
    axes[0].axis("off")

    axes[1].imshow(pred_slice.T, cmap="gray", origin="lower")
    axes[1].set_title("HR Reconstruction")
    axes[1].axis("off")

    axes[2].imshow(gt_slice.T, cmap="gray", origin="lower")
    axes[2].set_title("HR Reference")
    axes[2].axis("off")

    fig.savefig(file_path)


def evaluate(pred_mri, gt_mri):
    pred_slice = pred_mri.data[pred_mri.data.shape[0] // 2, :, :]
    gt_slice = gt_mri.data[gt_mri.data.shape[0] // 2, :, :]

    psnr_score = psnr(gt_slice, pred_slice, data_range=gt_slice.max() - gt_slice.min())
    ssim_score = ssim(gt_slice, pred_slice, data_range=gt_slice.max() - gt_slice.min())

    logging.info(f'PSNR: {psnr_score}, SSIM: {ssim_score}')
    
    return
