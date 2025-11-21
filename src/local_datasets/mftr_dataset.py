from local_distributions import MFTRDistribution
from typing import Optional
import torch


class MftrDataset:
    """
    Minimal PyTorch-like dataset for MFTR variates.
    If you want a strict torch.utils.data.Dataset subclass, change the class definition
    to inherit torch.utils.data.Dataset and import torch at top-level.
    """

    def __init__(
        self,
        num_samples: int,
        m: int = 8,
        mu: int = 7,
        K: float = 8.0,
        delta: float = 0.9,
        omega: float = 2.0,
        dist_type: str = "amplitude",
        seed: Optional[int] = None,
    ):
        """
        Args:
            num_samples: number of samples in the dataset
            m, mu, K, delta, omega: MFTR parameters (same meaning as original code)
            dist_type: 'amplitude' or 'power'
            seed: optional RNG seed for reproducibility
        """

        self.num_samples = int(num_samples)
        self.m = int(m)
        self.mu = int(mu)
        self.K = float(K)
        self.delta = float(delta)
        self.omega = float(omega)
        if dist_type not in ("amplitude", "power"):
            raise ValueError("dist_type must be 'amplitude' or 'power'")
        self.dist_type = dist_type
        self.seed = seed

        # SciPy's rvs random_state compatibility varies by version; we pass seed as int for safety
        mftr_gen = MFTRDistribution(
            name="mftr", n_inverse_terms=2000, dist_type=self.dist_type
        )

        samples = mftr_gen.rvs(
            self.m,
            self.K,
            self.delta,
            self.mu,
            self.omega,
            size=self.num_samples,
            random_state=seed,
        )

        self.samples = torch.tensor(samples, dtype=torch.float32).reshape(
            self.num_samples, 1
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx]
