import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


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


def save_reconstruction_comparison(pred_slice, gt_slice, file_path="images/comparison.png"):
    logging.info("Saving comparison to images/comparison.png")
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Display each slice
    axes[0].imshow(pred_slice.T, cmap="gray", origin="lower")
    axes[0].set_title("Pred Slice")
    axes[0].axis("off")

    axes[1].imshow(gt_slice.T, cmap="gray", origin="lower")
    axes[1].set_title("GT Slice")
    axes[1].axis("off")

    fig.savefig(file_path)
