from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from local_distributions import mftr


class MftrUniformConditionalDataset(Dataset):
    """MFTR conditional dataset with uniformly sampled parameter combinations.

    This dataset first samples `combos` parameter combinations uniformly within the
    provided ranges, then generates `samples_per_combo` MFTR samples for each
    combination.

    Conditions are stored efficiently as a tensor of shape (combos, 5) with the
    fixed order: [m, mu, K, delta, omega]. For a sample at index `idx`, the
    corresponding condition row is retrieved by `idx // samples_per_combo`.
    """

    def __init__(
        self,
        *,
        samples_per_combo: int,
        combos: int,
        m: tuple[float, float] = (8.0, 8.0),
        mu: tuple[float, float] = (7.0, 7.0),
        K: tuple[float, float] = (8.0, 8.0),
        delta: tuple[float, float] = (0.9, 0.9),
        omega: tuple[float, float] = (2.0, 2.0),
        dist_type: str = "amplitude",
        seed: Optional[int] = None,
        conds_raw: Optional[torch.Tensor] = None,
    ):
        self.samples_per_combo = int(samples_per_combo)
        if self.samples_per_combo <= 0:
            raise ValueError("samples_per_combo must be > 0")
        self.combos = int(combos)
        if conds_raw is None and self.combos <= 0:
            raise ValueError("combos must be > 0")

        if dist_type not in ("amplitude", "power"):
            raise ValueError("dist_type must be 'amplitude' or 'power'")
        self.dist_type = dist_type
        self.seed = seed

        if conds_raw is not None:
            if not isinstance(conds_raw, torch.Tensor):
                raise TypeError("conds_raw must be a torch.Tensor")
            if conds_raw.ndim != 2 or conds_raw.shape[1] != 5:
                raise ValueError("conds_raw must have shape (combos, 5)")
            self.conds_raw = conds_raw.to(dtype=torch.float32).contiguous()
            self.combos = int(self.conds_raw.shape[0])
        else:
            m_low, m_high = float(m[0]), float(m[1])
            mu_low, mu_high = float(mu[0]), float(mu[1])
            K_low, K_high = float(K[0]), float(K[1])
            delta_low, delta_high = float(delta[0]), float(delta[1])
            omega_low, omega_high = float(omega[0]), float(omega[1])

            if m_low > m_high:
                raise ValueError(f"Invalid range for 'm': {m}")
            if mu_low > mu_high:
                raise ValueError(f"Invalid range for 'mu': {mu}")
            if K_low > K_high:
                raise ValueError(f"Invalid range for 'K': {K}")
            if delta_low > delta_high:
                raise ValueError(f"Invalid range for 'delta': {delta}")
            if omega_low > omega_high:
                raise ValueError(f"Invalid range for 'omega': {omega}")

            if m_low < 1.0:
                raise ValueError("'m' must be >= 1")
            if mu_low < 1.0:
                raise ValueError("'mu' must be >= 1")
            if K_low < 0.0:
                raise ValueError("'K' must be >= 0")
            if omega_low <= 0.0:
                raise ValueError("'omega' must be > 0")
            if not (0.0 <= delta_low and delta_high < 1.0):
                raise ValueError("'delta' range must satisfy 0 <= low <= high < 1")

            rng = np.random.default_rng(seed)

            conds = np.empty((self.combos, 5), dtype=np.float32)
            for i in range(self.combos):
                m_i = m_low if m_low == m_high else float(rng.uniform(m_low, m_high))
                mu_i = (
                    mu_low
                    if mu_low == mu_high
                    else float(rng.uniform(mu_low, mu_high))
                )
                K_i = K_low if K_low == K_high else float(rng.uniform(K_low, K_high))
                delta_i = (
                    delta_low
                    if delta_low == delta_high
                    else float(rng.uniform(delta_low, delta_high))
                )
                omega_i = (
                    omega_low
                    if omega_low == omega_high
                    else float(rng.uniform(omega_low, omega_high))
                )

                conds[i] = np.array(
                    [m_i, mu_i, K_i, delta_i, omega_i], dtype=np.float32
                )

            self.conds_raw = torch.tensor(conds, dtype=torch.float32)

        mftr_gen = mftr(name="mftr", n_inverse_terms=2000, dist_type=self.dist_type)

        samples_all = []
        for i in range(self.combos):
            m_i, mu_i, K_i, delta_i, omega_i = (float(x) for x in self.conds_raw[i])
            s = mftr_gen.rvs(
                m_i,
                K_i,
                delta_i,
                mu_i,
                omega_i,
                size=self.samples_per_combo,
                random_state=None if seed is None else int(seed) + i,
            )
            samples_all.append(torch.tensor(s, dtype=torch.float32).reshape(-1, 1))

        self.samples = torch.cat(samples_all, dim=0)

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, idx: int):
        combo_idx = idx // self.samples_per_combo
        return self.samples[idx], self.conds_raw[combo_idx]
