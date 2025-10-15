import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.stats import nakagami


class NakagamiDataset(Dataset):
    def __init__(self, num_samples, nu=1.0, scale=1.0, seed: int | None = None):
        """
        Initializes the NakagamiDataset.

        Args:
            num_samples (int): The number of samples per signal.
            nu (float): The shape parameter of the Nakagami distribution.
                        Must be >= 0.5. Higher values indicate less severe fading.
            scale (float): The scale parameter of the Nakagami distribution.
                           Corresponds to the standard deviation of the underlying
                           Gaussian components.
            seed (int | None): Seed for the random number generator for reproducibility.
        """
        self.num_samples = num_samples
        self.nu = nu
        self.scale = scale
        self.samples = self.generate_nakagami_samples(seed)

    def generate_nakagami_samples(self, seed: int | None):
        """
        Generates samples from the Nakagami distribution.
        """
        rng = np.random.default_rng(seed)
        return torch.tensor(
            nakagami.rvs(
                nu=self.nu, scale=self.scale, size=self.num_samples, random_state=rng
            ),
            dtype=torch.float32,
        ).reshape(self.num_samples, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx]
