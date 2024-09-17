import torch
from Dataset import EdgeRadDataSet
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Network import AutoEncoder

"""
This script is used to train an autoencoder for a specific camera.
"""


def train_epoch(network, train_loader, loss_fn, optimiser, writer, epoch,
                device="cuda:0"):
    """
    Train the model for one epoch
    :param network: Model to be trained
    :param train_loader: Training data loader
    :param loss_fn: Loss function
    :param optimiser: Optimiser
    :param writer: Tensorboard writer
    :param epoch: Epoch number
    :param device: Device to train on
    """

    network.train()
    losses = []
    with tqdm(total=len(train_loader), dynamic_ncols=True) as tq:
        tq.set_description(
            f"Train :: Epoch: {epoch} ")

        for i, image in enumerate(train_loader):

            # Move to device
            image = image.to(device)

            # Forward pass
            latent = network.encoder(image)
            image_pred = network.decoder(latent)
            loss = loss_fn(image_pred, image)

            # Backward pass
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            losses.append(loss.item())
            tq.set_postfix(loss=loss.item())
            tq.update(1)

            # Write to tensorboard
            writer.add_scalar("Loss/train", loss.item(),
                              len(train_loader) * epoch + i)

        mean_loss = sum(losses) / len(losses)
        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")

    return mean_loss


@torch.no_grad()
def test_epoch(network, test_loader, writer, epoch, device):
    """
    Test the model for one epoch
    :param network: Model to be tested
    :param test_loader: Test data loader
    :param writer: Tensorboard writer
    :param epoch: Epoch number
    :param device: Device to test on
    """

    network.eval()
    losses = []
    with tqdm(total=len(test_loader), dynamic_ncols=True) as tq:
        tq.set_description(
            f"Test :: Epoch: {epoch} ")

        for i, image in enumerate(test_loader):

            # Move to device
            image = image.to(device)

            # Forward pass
            latent = network.encoder(image)
            image_pred = network.decoder(latent)

            # Calculate loss
            loss = torch.nn.MSELoss()(image_pred, image)

            losses.append(loss.item())

            tq.set_postfix(loss=loss.item())
            tq.update(1)

            # Plot the first image
            if i == 0:
                fig, ax = plt.subplots(1, 2)
                # First plot the real and predict after de-normalising
                ax[0].imshow((image[0, 0].cpu() * test_loader.dataset.dataset.image_std
                              + test_loader.dataset.dataset.image_mean).numpy())
                ax[1].imshow((image_pred[0, 0].cpu() * test_loader.dataset.dataset.image_std
                              + test_loader.dataset.dataset.image_mean).numpy())
                writer.add_figure("Test/Image", fig, epoch)
                # Now plot the same only normalised
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(image[0, 0].detach().cpu().numpy())
                ax[1].imshow(image_pred[0, 0].detach().cpu().numpy())
                writer.add_figure("Test/Image_norm", fig, epoch)

        mean_loss = sum(losses) / len(losses)
        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")
        writer.add_scalar("Loss/test", mean_loss, epoch)

    return mean_loss


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the network
    network = AutoEncoder([1, 32, 32, 64, 64], [2, 2, 2, 2],
                          [4096, 256, 32], 5).to(device)

    # Define the dataset nd train / test split
    full_dataset = EdgeRadDataSet("./Data/B1B2.h5")
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512,
                                              shuffle=True)

    # MSE loss and adam optimizer
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(network.parameters(), lr=1e-3)
    wrtier = SummaryWriter("./run")

    epochs = 1000
    test_losses = []
    for epoch in range(0, epochs):
        train_epoch(network, train_loader, loss_fn, optim, wrtier, epoch,
                    device)
        if epoch % 5 == 0:
            loss = test_epoch(network, test_loader, wrtier, epoch, device)

            test_losses.append(loss)

            if loss <= min(test_losses):
                torch.save(network.state_dict(),
                           f"./model_B1B2/model_{epoch}.pth")
                print(f"Model saved at epoch {epoch}")
