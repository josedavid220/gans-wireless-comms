from __future__ import annotations

from itertools import product
from typing import Optional

import torch
from torch.utils.data import Dataset

from local_distributions import mftr


class MftrConditionalDataset(Dataset):
    def __init__(
        self,
        samples_per_combo: int,
        param_grid: dict[str, list] | None = None,
        m: int = 8,
        mu: int = 7,
        K: float = 8.0,
        delta: float = 0.9,
        omega: float = 2.0,
        seed: Optional[int] = None,
    ):
        self.samples_per_combo = int(samples_per_combo)
        self.seed = seed

        defaults = {
            "m": int(m),
            "mu": int(mu),
            "K": float(K),
            "delta": float(delta),
            "omega": float(omega),
        }
        grid = param_grid or {}

        keys = [k for k in ("m", "mu", "K", "delta", "omega") if k in grid]
        values = [grid[k] for k in keys]

        combos: list[dict[str, float]] = []
        if keys:
            for vals in product(*values):
                combo = dict(defaults)
                for k, v in zip(keys, vals):
                    combo[k] = float(v)
                combos.append(combo)
        else:
            combos = [dict({k: float(v) for k, v in defaults.items()})]

        self.conds_raw = torch.tensor(
            [[c["m"], c["mu"], c["K"], c["delta"], c["omega"]] for c in combos],
            dtype=torch.float32,
        )

        mftr_gen = mftr(name="mftr", n_inverse_terms=2000, dist_type="amplitude")

        samples_all = []
        for i, c in enumerate(combos):
            s = mftr_gen.rvs(
                int(c["m"]),
                float(c["K"]),
                float(c["delta"]),
                int(c["mu"]),
                float(c["omega"]),
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
