import logging
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from data import get_train_dataloader, get_test_dataloader
from mri import MRI
from nn import NeuralNet, FourierMappingFactory
from utils import get_device, init_logging, read_config, save_reconstruction_comparison


init_logging()


def train(mri, num_epochs, batch_size, fourier_mapping, net, device):
    dataloader = get_train_dataloader(mri, batch_size=batch_size, device=device)

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-4)  # Adjust learning rate as needed

    # Loss function
    mse_loss = nn.MSELoss()

    # Number of epochs - one epoch means the model has seen all the data once
    num_epochs = num_epochs

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


def reconstruct_hr_mri(voxel_size, fourier_mapping, net, device):
    voxel_size = voxel_size
    voxel_loader = get_test_dataloader(voxel_size, 2048, device)

    # Set the network to evaluation mode
    net.eval()

    # Store predictions for each batch
    batch_predictions = []

    logging.info('Reconstructing high-resolution MRI')
    # Disable gradient calculations
    with torch.no_grad():
        for batch in voxel_loader:
            # Move batch to device
            coords = batch.to(device)
            coords = fourier_mapping.map(coords)

            # Predict intensity values for the batch
            batch_pred = net(coords)

            # Move predictions to CPU and store
            batch_predictions.append(batch_pred.cpu())

    # Concatenate all batch predictions
    all_predictions = torch.cat(batch_predictions, dim=0)

    # Convert to numpy array
    predicted_intensities_np = all_predictions.squeeze().numpy()

    # Reshape the array to the desired MRI dimensions
    x_dim, y_dim, z_dim = voxel_size
    predicted_mri = predicted_intensities_np.reshape((x_dim, y_dim, z_dim))

    pred_mri = MRI.from_data_and_voxel_size(predicted_mri, (x_dim, y_dim, z_dim))

    logging.info('Finished reconstruction')

    return pred_mri


def evaluate(pred_mri, gt_mri):
    pred_slice = pred_mri.data[pred_mri.data.shape[0] // 2, :, :]
    gt_slice = gt_mri.data[gt_mri.data.shape[0] // 2, :, :]

    psnr_score = psnr(gt_slice, pred_slice, data_range=gt_slice.max() - gt_slice.min())
    ssim_score = ssim(gt_slice, pred_slice, data_range=gt_slice.max() - gt_slice.min())

    logging.info(f'PSNR: {psnr_score}, SSIM: {ssim_score}')
    
    return


def main():
    # Training
    lr_mri_path = '../../hcp1200/996782/T1w_acpc_dc_restore_brain_downsample_factor_4.nii.gz'
    lr_mri = MRI.from_nii_file(lr_mri_path)

    config = read_config()
    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]
    eval_existing_model = config["training"]["eval_existing_model"]
    encoding = config["encoding"]
    input_size = config[encoding]["input_size"]
    scale = config[encoding]["scale"]
    voxel_size = config["reconstruction"]["voxel_size"]

    device = get_device()
    fourier_mapping = FourierMappingFactory.create(encoding, input_size=input_size, scale=scale, device=device)
    net = NeuralNet(input_size=input_size).to(device)

    if not eval_existing_model:
        train(lr_mri, num_epochs, batch_size, fourier_mapping, net, device)
        net.save_weights(f"weights/model_{num_epochs}.pth")
    else:
        net.load_weights(f"weights/model_{num_epochs}.pth")

    # Evaluation
    hr_mri_path = '../../hcp1200/996782/T1w_acpc_dc_restore_brain_downsample_factor_2.nii.gz'
    hr_mri = MRI.from_nii_file(hr_mri_path)

    pred_mri = reconstruct_hr_mri(hr_mri.data.shape, fourier_mapping, net, device)
    evaluate(pred_mri, hr_mri)
    save_reconstruction_comparison(lr_mri, pred_mri, hr_mri)


if __name__ == "__main__":
    main()
