import numpy as np
import torch
from Dataset import PredictDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Network import FullyConnected, AutoEncoder
from Losses import EmitLoss

"""
Script to train the predictor network.
"""


def train_epoch(predict_network, encoder_networks, train_loader, loss_fn,
                optimizer, writer, epoch, device):
    """
    Train the model for one epoch
    :param predict_network: Model to be trained
    :param encoder_networks: Encoder networks
    :param train_loader: Training data loader
    :param loss_fn: Loss function
    :param optimizer: Optimizer
    :param writer: Tensorboard writer
    :param epoch: Epoch number
    :param device: Device to train on
    """

    predict_network.train()
    encoder_networks[0].train()
    encoder_networks[1].train()
    encoder_networks[2].train()

    total_loss = []
    with (tqdm(total=len(train_loader), dynamic_ncols=True) as tq):
        tq.set_description(f"Train :: Epoch: {epoch} ")
        for i, (image_B1B2, image_B2B3, image_B3B4, size, field, _
                ) in enumerate(train_loader):

            # Calculate the ground truth beam parameters
            size = size.to(device)**2.
            field = field.to(device)
            _, beam_y_true, _, emit_y2_true = loss_fn.ground_truth_scan_batch(
                size, field)
            beam_y_true = torch.tile(beam_y_true, (20, 1, 1)
                                     ).transpose(0, 1).flatten(0, 1)
            emit_y2_true = torch.tile(emit_y2_true, (20, 1)
                                      ).transpose(0, 1).flatten(0, 1)

            # Get the predicted beam parameters, and flatten to pass through
            # the network
            size = size.flatten(0, 1)
            field = field.flatten(0, 1)[:, None]
            image_B1B2 = image_B1B2.to(device).flatten(0, 1)[:, None]
            image_B2B3 = image_B2B3.to(device).flatten(0, 1)[:, None]
            image_B3B4 = image_B3B4.to(device).flatten(0, 1)[:, None]
            latent_b1b2 = encoder_networks[0].encoder(image_B1B2)
            latent_b2b3 = encoder_networks[1].encoder(image_B2B3)
            latent_b3b4 = encoder_networks[2].encoder(image_B3B4)
            # append ythe field strength to the vector
            input_params = torch.cat(
                (latent_b1b2, latent_b2b3, latent_b3b4), dim=1)
            input_params = torch.cat((field, input_params), dim=1)
            beam_params_predict = predict_network(input_params)

            # Make sure the beam size and divergence are positive
            beam_size_y = torch.abs(beam_params_predict[..., 0])
            beam_corr_y = beam_params_predict[..., 1]
            beam_div_y = torch.abs(beam_params_predict[..., 2])
            beam_params_predict = torch.stack([
                beam_size_y, beam_corr_y, beam_div_y]).T

            # Calculate the beam parameters through quad scan then propagating
            # forward through the beamline
            beam_y_true = loss_fn.propogae_forward(field[:, 0], beam_y_true)
            emit_y2_pred = loss_fn.get_single_emittance(beam_params_predict)

            # Calculate the loss First is based on measured size, second are the
            # coor and div
            loss_1 = torch.mean((beam_params_predict[:, 0] - size[:, 1])**2.)
            loss_2 = torch.mean((beam_params_predict[:, 1:] - beam_y_true[:, 1:]
                                 )**2.)

            # Adding an additional loss on the emittance can help
            loss_3 = torch.mean((emit_y2_pred - emit_y2_true)**2.)
            loss = loss_1 + loss_2 + 10 * loss_3

            # Now do the standard backpass and optimiser step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss.append(loss.item())
            tq.set_postfix(loss=loss.item())
            tq.update()

        total_loss = sum(total_loss) / len(total_loss)
        writer.add_scalar("Train/Total", total_loss, epoch)

    return total_loss


@torch.no_grad()
def test_epoch(predict_network, encoder_networks, test_loader, loss_fn,
               emit_y, writer, epoch, device):
    """
    :param predict_network: The prediction network
    :param encoder_networks: The encoder networks
    :param test_loader: The test data loader
    :param loss_fn: The loss function mse
    :param emit_y: The mean emittance for the test data
    :param writer: The tensorboard writer
    :param epoch: The epoch number
    :param device: Device to train on
    """

    # Set the networks to evaluation mode
    predict_network.eval()
    encoder_networks[0].eval()
    encoder_networks[1].eval()
    encoder_networks[2].eval()
    losses = []

    with tqdm(total=len(test_loader), dynamic_ncols=True) as tq:

        fig1, ax1 = plt.subplots(1, 3, figsize=(12, 3))
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        loss_emit = 0
        loss_size = 0
        emit_pred = []
        emit_true = []

        for i, (image_B1B2, image_B2B3, image_B3B4, size, field, batch_index
                ) in enumerate(test_loader):

            # Calculate the ground truth beam parameters
            size = size.to(device)**2.
            field = field.to(device)
            mean_field = torch.mean(field, dim=0)
            _, beam_y_true, _, emit_y2_true = loss_fn.ground_truth_scan_batch(
                size, field)

            size = size.flatten(0, 1)
            field = field.flatten(0, 1)[:, None]
            image_B1B2 = image_B1B2.to(device).flatten(0, 1)[:, None]
            image_B2B3 = image_B2B3.to(device).flatten(0, 1)[:, None]
            image_B3B4 = image_B3B4.to(device).flatten(0, 1)[:, None]
            latent_b1b2 = encoder_networks[0].encoder(image_B1B2)
            latent_b2b3 = encoder_networks[1].encoder(image_B2B3)
            latent_b3b4 = encoder_networks[2].encoder(image_B3B4)
            input_params = torch.cat(
                (latent_b1b2, latent_b2b3, latent_b3b4), dim=1)
            input_params = torch.cat((field, input_params), dim=1)
            beam_params_predict = predict_network(input_params)

            # Make sure the beam size and divergence are positive
            beam_size_y = torch.abs(beam_params_predict[..., 0])
            beam_corr_y = beam_params_predict[..., 1]
            beam_div_y = torch.abs(beam_params_predict[..., 2])
            beam_params_predict = torch.stack([
                beam_size_y, beam_corr_y, beam_div_y]).T
            emit_y2_pred = loss_fn.get_single_emittance(
                beam_params_predict)

            #  Plot of beam size**2 vs field (16.5um pixel size)
            ax1[0].scatter(field[:, 0].cpu(), beam_size_y.cpu()
                           * (test_loader.dataset.moments_norm[1] * 16.5)**2.
                           , color="red", marker="x")
            ax1[1].scatter(field[:, 0].cpu(), beam_corr_y.cpu()
                           * (test_loader.dataset.moments_norm[1] * 16.5)**2.,
                           color="red", marker="x")
            ax1[2].scatter(field[:, 0].cpu(), beam_div_y.cpu()
                           * (test_loader.dataset.moments_norm[1] * 16.5)**2.,
                           color="red", marker="x")

            # plot the predicted vs true beam size
            ax2.scatter(size[:, 1].cpu()**0.5
                        * test_loader.dataset.moments_norm[1] * 16.5,
                        beam_size_y.cpu()**0.5
                        * test_loader.dataset.moments_norm[1] * 16.5,
                        color="red", marker="x")

            # Calculate val losses
            emit_y2_true = torch.tile(emit_y2_true, (20, 1)).T.flatten(0, 1)
            loss_emit += torch.mean((emit_y2_true - emit_y2_pred)**2.)
            loss_size += torch.mean((size[:, 1] - beam_size_y)**2.)
            emit_pred.append(emit_y2_pred)
            emit_true.append(emit_y2_true)

        # Set the axis labels and what not
        ax1[0].set_xlabel("k " + "$(1/m^2)$")
        ax1[0].set_ylabel("$\Sigma_{11}$" + "$\,\, (\mu m^2)$")

        ax1[1].set_xlabel("k  " + "$(1/m^2)$")
        ax1[1].set_ylabel("$\Sigma_{12}$" + "$\,\, (\mu m\,\, \mu rad)$")

        ax1[2].set_xlabel("k  " + "$(1/m^2)$")
        ax1[2].set_ylabel("$\Sigma_{22}$" + "$\,\, ({\mu rad}^2)$")
        fig1.tight_layout()

        ax2.plot([150, 450], [150, 450], color="blue")
        ax2.set_xlabel("$\sigma_y$" + " true" + "$\,\, (\mu m)$")
        ax2.set_ylabel("$\sigma_y$" + " predict" + "$\,\, (\mu m)$")
        ax2.set_xlim([150, 450])
        ax2.set_ylim([150, 450])
        fig2.tight_layout()

        loss_emit /= len(test_loader)
        loss_size /= len(test_loader)

        # Now do the emittance scan
        emit_loss_test = 0
        fig4, ax4 = plt.subplots(figsize=(4, 3))
        B1B2_test, B2B3_test, B3B4_test, size_test, field_test \
            = test_loader.dataset.get_test()
        for i, (b1b2, b2b3, b3b4, size, field) in enumerate(
                zip(B1B2_test, B2B3_test, B3B4_test, size_test, field_test)):
            b1b2, b2b3, b3b4, field = (b1b2.to(device)[:, None],
                                       b2b3.to(device)[:, None],
                                       b3b4.to(device)[:, None],
                                       field.to(device))

            latent_b1b2 = encoder_networks[0].encoder(b1b2)
            latent_b2b3 = encoder_networks[1].encoder(b2b3)
            latent_b3b4 = encoder_networks[2].encoder(b3b4)
            input_params = torch.cat((latent_b1b2, latent_b2b3,
                                      latent_b3b4), dim=1)
            input_params = torch.cat((field[:, None], input_params), dim=1)

            beam_params_predict = predict_network(input_params)
            beam_size_y = torch.abs(beam_params_predict[:, 0])
            beam_corr_y = beam_params_predict[:, 1]
            beam_div_y = torch.abs(beam_params_predict[:, 2])
            beam_params_predict = torch.stack(
                [beam_size_y, beam_corr_y, beam_div_y]).T
            emity2_predict = loss_fn.get_single_emittance(beam_params_predict)
            emity2_predict = torch.where(emity2_predict < 0., 1.e-9, emity2_predict)

            # Convert emittance back to normal units
            emit_pred = 10.2 + 4.7 * (emity2_predict.cpu().numpy()**0.5
                                      - 0.6110) / 0.2595
            emit_true = 10.2 + 4.7 * (torch.tile(emit_y[i], (20, 1)
                                        )[:, 0].cpu()**0.5 - 0.6110) / 0.2595

            # Plot the emittance for each batch
            ax4.scatter(np.mean(emit_true.numpy()), np.mean(emit_pred),
                        color="red", marker="x")
            ax4.errorbar(np.mean(emit_true.numpy()), np.mean(emit_pred),
                         yerr=np.std(emit_pred), capsize=4., color="red")
            emit_loss_test += torch.mean((torch.tile(emit_y[i], (20, 1))[:, 0]
                                          - emity2_predict)**2.)

        # Set the axis labels
        ax4.plot([10, 15], [10, 15], color="blue")
        ax4.set_xlabel("$\epsilon_y $" + " true " + "$\,\, (\mu m)$")
        ax4.set_ylabel("$\epsilon_y $" + " predict " + "$\,\, (\mu m)$")
        fig4.tight_layout()

        # Write figures to tensorboard
        writer.add_figure("Test/BeamSize", fig1, epoch)
        writer.add_figure("Test/Emittance", fig2, epoch)
        writer.add_figure("Test/emit hist", fig3, epoch)
        writer.add_figure("Test/EmittanceScan", fig4, epoch)

        # Write metrics
        writer.add_scalar("Test/Size_loss", loss_size, epoch)
        writer.add_scalar("Test/emit_loss_main", emit_loss_test, epoch)


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Define the autoencoder networks
    net_b1b2 = AutoEncoder([1, 32, 32, 64, 64], [2, 2, 2, 2],
                          [4096, 256, 32], 5).to(device)
    net_b2b3 = AutoEncoder([1, 32, 32, 64, 64], [2, 2, 2, 2],
                          [4096, 256, 32], 5).to(device)
    net_b3b4 = AutoEncoder([1, 32, 32, 64, 64], [2, 2, 2, 2],
                          [4096, 256, 32], 5).to(device)

    predict_network = FullyConnected(16, 3).to(device)

    # Load the weights
    #net_b1b2.load_state_dict(torch.load("../AutoEncoder/model_B1B2/model_180.pth"))
    #net_b2b3.load_state_dict(torch.load("../AutoEncoder/model_B2B3/model_525.pth"))
    #net_b3b4.load_state_dict(torch.load("../AutoEncoder/model_B3B4/model_265.pth"))
    encoder_networks = [net_b1b2, net_b2b3, net_b3b4]

    # Define the dataset for training / testing
    dataset_train = PredictDataset("./Data/Train.h5")
    dataset_test = PredictDataset("./Data/Test.h5", test=True)
    dataset_test.set_mean_std(*dataset_train.get_mean_std())
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=4, shuffle=True)

    # Define the loss function
    r_mat_375_x = torch.tensor(
        [[-0.265156162290672, -0.479573440258417],
         [-0.741080911610350, -5.11171496291679]]).float().to(device)
    r_mat_375_y = torch.tensor(
        [[-1.08297699392360, -3.08358368551067],
         [-0.468711376539411, -2.25795263207843]]).float().to(device)
    r_mats_375 = [r_mat_375_y, r_mat_375_y]
    loss = EmitLoss(r_mats_375)

    writer = SummaryWriter(filename_suffix="test")
    optimiser = torch.optim.Adam(predict_network.parameters(), lr=1e-4)
    # Can also tune the
    #optimiser = torch.optim.Adam(list(predict_network.parameters())
    #                             + list(net_b1b2.parameters())
    #                             + list(net_b2b3.parameters())
    #                             + list(net_b3b4.parameters())
    #                             , lr=1e-4)

    # Calculate the mean emittance for the test and train
    field_total, size_total = dataset_train.get_full_field_size()
    field_total = torch.mean(field_total, dim=2).to(device)
    size_total_2 = torch.mean(size_total, dim=2).to(device)**2.
    _, _, _, emit_y = loss.ground_truth_scan_batch(size_total_2, field_total)

    for epoch in range(2000):
        loss_train = train_epoch(predict_network, encoder_networks,
                                 train_loader, loss, optimiser, writer,
                                 epoch, device)
        test_epoch(predict_network, encoder_networks, test_loader,
                   loss, emit_y, writer, epoch, device)
