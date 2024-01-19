import logging
import torch


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
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
