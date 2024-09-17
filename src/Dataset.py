import torch
import h5py
from torch.utils.data import Dataset
import numpy as np


class EdgeRadDataSet(Dataset):
    """
    Dataset for edge radiation image. See GenDataset_AE.ipynb for details
    """

    def __init__(self, path):
        self.file = h5py.File(path, "r")
        self.images = torch.from_numpy(np.array(self.file["Images"]))
        self.image_mean = torch.mean(self.images, dim=0)
        self.image_std = torch.std(self.images)

    def __len__(self):
        """
        :return: length of the dataset
        """
        return self.images.shape[0]

    def __getitem__(self, index):
        """
        :param index: index of dataset
        :return: Intensity image
        """
        image = self.images[index][None]
        image = (image - self.image_mean) / self.image_std
        return image


class PredictDataset(Dataset):

    def __init__(self, path, test=False):
        """
        :param path: Path to dataset
        :param test: If the dataset is for testing
        """
        self.file = h5py.File(path, "r")
        self.b1b2_images = torch.from_numpy(np.array(self.file["B1B2"])).float()
        self.b2b3_images = torch.from_numpy(np.array(self.file["B2B3"])).float()
        self.b3b4_images = torch.from_numpy(np.array(self.file["B3B4"])).float()
        self.moments = torch.from_numpy(np.array(self.file["Moments"])).float()
        self.field = torch.from_numpy(np.array(self.file["Field"])).float()

        self.b1b2_mean = torch.mean(self.b1b2_images, dim=(0, 1, 2))
        self.b2b3_mean = torch.mean(self.b2b3_images, dim=(0, 1, 2))
        self.b3b4_mean = torch.mean(self.b3b4_images, dim=(0, 1, 2))

        self.b1b2_std = torch.std(self.b1b2_images)
        self.b2b3_std = torch.std(self.b2b3_images)
        self.b3b4_std = torch.std(self.b3b4_images)

        self.moments_norm = torch.tensor([self.moments[..., 2].max() / 6.,
                                          self.moments[..., 3].max() / 6.])

        if test:
            self.b1b2_test = torch.from_numpy(np.array(self.file["/Sample/B1B2"])).float()
            self.b2b3_test = torch.from_numpy(np.array(self.file["/Sample/B2B3"])).float()
            self.b3b4_test = torch.from_numpy(np.array(self.file["/Sample/B3B4"])).float()
            self.moments_test = torch.from_numpy(np.array(self.file["/Sample/Moments"])).float()
            self.field_test = torch.from_numpy(np.array(self.file["/Sample/Field"])).float()

        self.n_batches = self.b1b2_images.shape[0]
        self.test = test

    def __len__(self):
        """
        :return: length of the dataset
        """
        return len(self.b1b2_images) * 20

    def __getitem__(self, index):
        """
        :param index: index of dataset
        """
        batch_index = np.random.randint(0, self.n_batches, 1)[0]
        n_steps = self.b1b2_images[batch_index].shape[0]
        n_steps = 20
        step_index = np.arange(6, 26)
        shot_index = np.random.randint(0, 40, n_steps)

        b1b2 = (self.b1b2_images[batch_index, step_index, shot_index] -
                self.b1b2_mean) / self.b1b2_std
        b2b3 = (self.b2b3_images[batch_index, step_index, shot_index] -
                self.b2b3_mean) / self.b2b3_std
        b3b4 = (self.b3b4_images[batch_index, step_index, shot_index] -
                self.b3b4_mean) / self.b3b4_std
        field = self.field[batch_index, step_index, shot_index]
        size = self.moments[batch_index, step_index, shot_index, 2:]
        size = size / self.moments_norm

        return b1b2, b2b3, b3b4, size, field, batch_index

    def set_mean_std(self, means, stds):
        """
        Set the mean and std of the dataset
        :param means: Mean of the dataset
        :param stds: Std of the dataset
        """
        self.b1b2_mean = means[0]
        self.b1b2_std = stds[0]
        self.b2b3_mean = means[1]
        self.b2b3_std = stds[1]
        self.b3b4_mean = means[2]
        self.b3b4_std = stds[2]
        self.moments_norm = means[3]

    def get_mean_std(self):
        means = [self.b1b2_mean, self.b2b3_mean, self.b3b4_mean,
                 self.moments_norm]
        stds = [self.b1b2_std, self.b2b3_std, self.b3b4_std]
        return means, stds

    def get_test(self):
        """
        Get the test dataset
        """
        if not self.test:
            raise ValueError("Dataset is not a test dataset")

        b1b2_test = (self.b1b2_test[:, 6:26] - self.b1b2_mean) / self.b1b2_std
        b2b3_test = (self.b2b3_test[:, 6:26] - self.b2b3_mean) / self.b2b3_std
        b3b4_test = (self.b3b4_test[:, 6:26] - self.b3b4_mean) / self.b3b4_std
        size_test = self.moments_test[:, 6:26, 2:] / self.moments_norm
        field_test = self.field_test[:, 6:26]
        return b1b2_test, b2b3_test, b3b4_test, size_test, field_test

    def get_full_field_size(self):
        """
        Get full scan data to get emittances
        :return:
        """
        field = self.field[:, 6:26, :]
        size = self.moments[:, 6:26, :, 2:] / self.moments_norm
        return field, size

