import logging
import torch


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
