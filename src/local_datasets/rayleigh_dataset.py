import torch
import numpy as np
from torch.utils.data import Dataset


class RayleighDataset(Dataset):
    def __init__(self, num_samples, scale=1.0):
        """
        Initializes the RayleighDataset.

        Args:
            num_samples (int): The number of samples per signal.
            scale (float): The scale parameter of the Rayleigh distribution.
                           Corresponds to the standard deviation of the underlying
                           Gaussian components.
            num_signals (int): The number of distinct signals to generate.
        """
        self.num_samples = num_samples
        self.scale = scale
        self.samples = self.generate_rayleigh_samples()

    def generate_rayleigh_samples(self):
        """
        Generates multiple Rayleigh-distributed signals.
        """
        rng = np.random.default_rng(0)
        return torch.tensor(
            rng.rayleigh(scale=self.scale, size=self.num_samples), dtype=torch.float32
        ).reshape(self.num_samples, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx]
