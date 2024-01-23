import logging
import torch.nn as nn
import torch.optim as optim

from data import get_train_dataloader, get_test_dataloader
from mri import MRI
from nn import NeuralNet, PosEncMapping, GaussianMapping
from utils import get_device, init_logging


init_logging()


def main():
    # Load the MRI data from the file
    file_path = '../../hcp1200/996782/T1w_acpc_dc_restore_brain_downsample_factor_8.nii.gz'
    mri = MRI(file_path)

    device = get_device()
    dataloader = get_train_dataloader(mri, batch_size=2048, device=device)

    # Initialize network
    encoding = 'pos_enc'
    if encoding == 'pos_enc':
        input_size = 120 #192 + 96
        fourier_mapping = PosEncMapping(num_frequencies=input_size//6, scale=1000)
    else:
        input_size = 256
        fourier_mapping = GaussianMapping(num_frequencies=input_size//2, scale=10, device=device)

    net = NeuralNet(input_size=input_size).to(device)

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-4)  # Adjust learning rate as needed

    # Loss function
    mse_loss = nn.MSELoss()

    # Number of epochs - one epoch means the model has seen all the data once
    num_epochs = 500  # Adjust as needed

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            # Get the input features and target labels, and put them on the device
            coords, gt_intensity_values = data
            coords, gt_intensity_values = coords.to(device), gt_intensity_values.to(device)
            coords = fourier_mapping.map(coords)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: compute predicted outputs by passing inputs to the model
            pred_intensity_values = net(coords)

            # Compute loss
            loss = mse_loss(pred_intensity_values.squeeze(), gt_intensity_values)

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Perform a single optimization step (parameter update)
            optimizer.step()

            # Print statistics
            running_loss += loss.item()

        logging.info(f'[{epoch + 1}] loss: {running_loss / 100:.3f}')
        running_loss = 0.0

    logging.info('Finished Training')


if __name__ == "__main__":
    main()