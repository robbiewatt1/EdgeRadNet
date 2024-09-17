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
