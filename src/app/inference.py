from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import torch
from scipy.stats import cramervonmises_2samp, ks_2samp, wasserstein_distance

from cgans import CGAN
from local_distributions import mftr
from local_datasets.mftr_uniform_conditional_dataset import normalize_conds_minus1_1

from .artifacts import VersionArtifacts
from .config import DEFAULT_BATCH_SIZE


def _validate_params(m: float, mu: float, K: float, delta: float, omega: float) -> str | None:
    if m < 1.0:
        return "m must be >= 1"
    if mu < 1.0:
        return "mu must be >= 1"
    if K < 0.0:
        return "K must be >= 0"
    if not (0.0 <= delta < 1.0):
        return "delta must satisfy 0 <= delta < 1"
    if omega <= 0.0:
        return "omega must be > 0"
    return None


@lru_cache(maxsize=8)
def load_model_cached(checkpoint_path: str) -> CGAN:
    model = CGAN.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    return model


@lru_cache(maxsize=2)
def mftr_sampler(dist_type: str):
    return mftr(name="mftr", n_inverse_terms=2000, dist_type=dist_type)


def _metrics_two_sample(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    n = min(len(x), len(y))
    xs = np.sort(x)[:n]
    ys = np.sort(y)[:n]

    ks = ks_2samp(x, y)
    cvm = cramervonmises_2samp(x, y)

    ks_stat = getattr(ks, "statistic", None)
    ks_pvalue = getattr(ks, "pvalue", None)
    if ks_stat is None or ks_pvalue is None:
        ks_stat, ks_pvalue = ks[0], ks[1]

    ks_stat_f = float(np.asarray(ks_stat).reshape(()).item())
    ks_pvalue_f = float(np.asarray(ks_pvalue).reshape(()).item())

    return {
        "n": float(n),
        "mae": float(np.mean(np.abs(xs - ys))),
        "mse": float(np.mean((xs - ys) ** 2)),
        "ks_stat": ks_stat_f,
        "ks_pvalue": ks_pvalue_f,
        "cvm_stat": float(cvm.statistic),
        "cvm_pvalue": float(cvm.pvalue),
        "wasserstein": float(wasserstein_distance(x, y)),
    }


def run_comparison(
    *,
    art: VersionArtifacts,
    m: float,
    mu: float,
    K: float,
    delta: float,
    omega: float,
    n_samples: int,
    seed: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], str | None]:
    err = _validate_params(m, mu, K, delta, omega)
    if err is not None:
        return np.array([]), np.array([]), {}, err

    if art.checkpoint is None:
        return np.array([]), np.array([]), {}, "No checkpoint found for selected model version"

    model = load_model_cached(str(art.checkpoint))
    ranges = art.ranges
    normalize_conds = bool(art.eval_metadata.get("normalize_conds", True))
    dist_type = str(art.eval_metadata.get("dist_type", "amplitude"))

    cond_raw = torch.tensor([[m, mu, K, delta, omega]], dtype=torch.float32)
    if normalize_conds:
        conds = normalize_conds_minus1_1(
            cond_raw,
            (
                ranges["m"],
                ranges["mu"],
                ranges["K"],
                ranges["delta"],
                ranges["omega"],
            ),
        )
    else:
        conds = cond_raw

    conds_rep = conds.repeat(int(n_samples), 1)

    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(int(seed))

    outputs = []
    with torch.no_grad():
        for start in range(0, int(n_samples), int(batch_size)):
            c = conds_rep[start : start + int(batch_size)].to(device)
            z = torch.randn(c.shape[0], int(model.latent_dim), device=device, generator=gen)
            x = model(z, c)
            outputs.append(x.detach().cpu())
    generated = torch.cat(outputs, dim=0).reshape(-1).numpy().astype(float)

    sampler = mftr_sampler(dist_type)
    real = sampler.rvs(
        float(m),
        float(K),
        float(delta),
        float(mu),
        float(omega),
        size=int(n_samples),
        random_state=int(seed),
    )
    real = np.asarray(real, dtype=float).reshape(-1)

    metrics = _metrics_two_sample(real, generated)
    return real, generated, metrics, None


def metrics_table(metrics: dict[str, float]) -> list[list[Any]]:
    if not metrics:
        return []
    return [
        ["MAE (quantiles)", metrics["mae"], "Average absolute quantile gap"],
        ["MSE (quantiles)", metrics["mse"], "Average squared quantile gap"],
        ["KS statistic", metrics["ks_stat"], "Maximum ECDF distance"],
        ["KS p-value", metrics["ks_pvalue"], "Significance for KS test"],
        ["CvM statistic", metrics["cvm_stat"], "Integrated ECDF discrepancy"],
        ["CvM p-value", metrics["cvm_pvalue"], "Significance for CvM test"],
        ["Wasserstein", metrics["wasserstein"], "Earth mover distance (1D)"],
    ]
