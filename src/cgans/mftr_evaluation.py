from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any
from pathlib import Path

import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import cramervonmises_2samp, ks_2samp, wasserstein_distance

from local_distributions import mftr
from local_distributions.mftr_dist import pdf_mftr

from local_datasets.mftr_uniform_conditional_dataset import normalize_conds_minus1_1


# -----------------------
# Plot styling (IEEE two-column friendly)
# -----------------------


IEEE_COL_IN = 3.5
IEEE_DOUBLE_COL_IN = 7.16


def _maybe_apply_ieee_style() -> None:
    """Apply repo-wide IEEE Matplotlib style if available."""

    try:
        from ieee_plot_style import apply_ieee_style, find_repo_root

        apply_ieee_style(find_repo_root(Path(__file__).resolve()))
    except Exception:
        # If style isn't available for any reason, keep defaults.
        return


@dataclass(frozen=True)
class MftrRanges:
    m: tuple[float, float]
    mu: tuple[float, float]
    K: tuple[float, float]
    delta: tuple[float, float]
    omega: tuple[float, float]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _to_1d_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x).reshape(-1)
    return x.astype(np.float64, copy=False)


def _float_scalar(v: Any) -> float:
    """Coerce a scalar-like value to a Python float (robust to NumPy scalars)."""
    return float(np.asarray(v).reshape(()).item())


def _empirical_qq(ax, x: np.ndarray, y: np.ndarray, title: str) -> None:
    n = min(len(x), len(y))
    if n <= 1:
        ax.set_title(title)
        return
    xs = np.sort(x)[:n]
    ys = np.sort(y)[:n]
    ax.scatter(xs, ys, s=4, alpha=0.35)
    lo = float(min(xs.min(), ys.min()))
    hi = float(max(xs.max(), ys.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Real quantiles")
    ax.set_ylabel("Generated quantiles")
    ax.grid(True, alpha=0.2)


def _hist_kde(ax, x: np.ndarray, y: np.ndarray, title: str) -> None:
    sns.kdeplot(x, ax=ax, label="Real", color="black", linewidth=2)
    sns.kdeplot(y, ax=ax, label="Generated", color="tab:blue", linewidth=2)
    sns.histplot(x, ax=ax, stat="density", color="black", alpha=0.15, bins=60)
    sns.histplot(y, ax=ax, stat="density", color="tab:blue", alpha=0.15, bins=60)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)


def _generated_hist_kde_vs_theoretical_pdf(
    ax,
    gen: np.ndarray,
    *,
    m: float,
    mu: float,
    K: float,
    delta: float,
    omega: float,
    title: str,
    show_legend: bool,
) -> None:
    # Generated distribution (empirical)
    sns.histplot(gen, ax=ax, stat="density", color="tab:blue", alpha=0.15, bins=60)
    sns.kdeplot(gen, ax=ax, label="Generated KDE", color="tab:blue", linewidth=2)

    # Theoretical MFTR PDF curve.
    # Match the x_range definition used in mftr_plotting.py
    x_range = np.linspace(1e-6, 3.0 * np.sqrt(float(omega)), 1000)
    theoretical_pdf = pdf_mftr(
        x_range**2,
        m=int(m),
        K=float(K),
        delta=float(delta),
        mu=int(mu),
        omega=float(omega),
    )
    theoretical_pdf = theoretical_pdf * (2.0 * x_range)
    ax.plot(x_range, theoretical_pdf, "r-", linewidth=2, label="MFTR PDF")

    ax.set_title(title)
    if show_legend:
        ax.legend()
    ax.grid(True, alpha=0.2)


def _format_params(m: float, mu: float, K: float, delta: float, omega: float) -> str:
    # Keep this compact; long titles get unreadable quickly.
    return f"m={m:.3f}, mu={mu:.3f}, K={K:.3f}, δ={delta:.3f}, ω={omega:.3f}"


def _format_ranges(ranges: MftrRanges) -> str:
    def r(name: str, lo: float, hi: float) -> str:
        if lo == hi:
            return f"{name}={lo:g}"
        return f"{name}∈[{lo:g},{hi:g}]"

    return (
        "Train ranges: "
        + ", ".join(
            [
                r("m", *ranges.m),
                r("mu", *ranges.mu),
                r("K", *ranges.K),
                r("δ", *ranges.delta),
                r("ω", *ranges.omega),
            ]
        )
    )


def _two_sample_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    n = min(len(x), len(y))
    xs = np.sort(x)[:n]
    ys = np.sort(y)[:n]

    mse = float(np.mean((xs - ys) ** 2))
    mae = float(np.mean(np.abs(xs - ys)))
    ks = ks_2samp(x, y, alternative="two-sided")
    cvm = cramervonmises_2samp(x, y)
    w = float(wasserstein_distance(x, y))

    ks_stat = getattr(ks, "statistic", None)
    ks_pvalue = getattr(ks, "pvalue", None)
    if ks_stat is None or ks_pvalue is None:
        # Older SciPy may return a 2-tuple
        ks_stat = ks[0]
        ks_pvalue = ks[1]

    return {
        "n": int(n),
        "mse_quantiles": mse,
        "mae_quantiles": mae,
        "ks_stat": _float_scalar(ks_stat),
        "ks_pvalue": _float_scalar(ks_pvalue),
        "cvm_stat": float(cvm.statistic),
        "cvm_pvalue": float(cvm.pvalue),
        "wasserstein": w,
    }


def _generate_model_samples(
    *,
    model,
    conds: torch.Tensor,
    latent_dim: int,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    device = model.device
    model.eval()

    out = []
    gen = torch.Generator(device=device).manual_seed(int(seed))
    with torch.no_grad():
        for start in range(0, conds.shape[0], int(batch_size)):
            c = conds[start : start + int(batch_size)].to(device)
            z = torch.randn(c.shape[0], int(latent_dim), device=device, generator=gen)
            x = model(z, c)
            out.append(x.detach().cpu())
    return torch.cat(out, dim=0)


def _repeat_conds_per_sample(
    conds_raw: torch.Tensor, samples_per_combo: int
) -> torch.Tensor:
    return conds_raw.repeat_interleave(int(samples_per_combo), dim=0)


def sample_params_within(ranges: MftrRanges, n: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(int(seed))

    def u(a: float, b: float) -> float:
        return float(a if a == b else rng.uniform(a, b))

    conds = np.empty((int(n), 5), dtype=np.float32)
    for i in range(int(n)):
        conds[i] = np.array(
            [
                u(*ranges.m),
                u(*ranges.mu),
                u(*ranges.K),
                u(*ranges.delta),
                u(*ranges.omega),
            ],
            dtype=np.float32,
        )
    return torch.tensor(conds, dtype=torch.float32)


def sample_params_outside(ranges: MftrRanges, n: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(int(seed))

    def out_1d(
        low: float,
        high: float,
        *,
        min_v: float | None = None,
        max_v: float | None = None,
    ) -> float:
        width = float(high - low)
        if width == 0.0:
            width = max(1e-3, abs(high) * 0.25)

        # choose below or above with equal probability
        if rng.random() < 0.5:
            a, b = low - width, low
        else:
            a, b = high, high + width

        if min_v is not None:
            a, b = max(a, min_v), max(b, min_v)
        if max_v is not None:
            a, b = min(a, max_v), min(b, max_v)

        # if interval collapses due to clipping, fall back to boundary
        if not (a < b):
            return float(
                min(max(low, min_v) if min_v is not None else low, max_v)
                if max_v is not None
                else low
            )

        return float(rng.uniform(a, b))

    eps = 1e-6
    conds = np.empty((int(n), 5), dtype=np.float32)
    for i in range(int(n)):
        m = out_1d(*ranges.m, min_v=1.0)
        mu = out_1d(*ranges.mu, min_v=1.0)
        K = out_1d(*ranges.K, min_v=0.0)
        delta = out_1d(*ranges.delta, min_v=0.0, max_v=1.0 - eps)
        omega = out_1d(*ranges.omega, min_v=eps)
        conds[i] = np.array([m, mu, K, delta, omega], dtype=np.float32)

    return torch.tensor(conds, dtype=torch.float32)


def run_end_of_training_evaluation(
    *,
    out_dir: str,
    model,
    train_dataset,
    ranges: MftrRanges,
    dist_type: str,
    seed: int,
    eval_max_mixture_samples: int,
    eval_num_params_in: int,
    eval_num_params_out: int,
    eval_num_samples_per_param: int,
    normalize_conds: bool,
    batch_size: int,
    eval_save_per_experiment_images: bool,
    run_mixture_eval: bool = True,
) -> None:
    """Creates plots + JSON metrics under out_dir.

    - Mixture eval: compares real training mixture vs generated mixture using the
      same condition vector per sample.
    - Param eval: compares generated conditional samples vs theoretical MFTR samples
      for random in-range and out-of-range parameter sets.
    """

    _ensure_dir(out_dir)
    _maybe_apply_ieee_style()
    torch.save(
        train_dataset.conds_raw.detach().cpu(),
        os.path.join(out_dir, "train_conds_raw.pt"),
    )
    meta = {
        "seed": int(seed),
        "dist_type": str(dist_type),
        "ranges": asdict(ranges),
        "normalize_conds": bool(normalize_conds),
        "samples_per_combo": int(getattr(train_dataset, "samples_per_combo", -1)),
        "combos": int(
            getattr(train_dataset, "combos", train_dataset.conds_raw.shape[0])
        ),
        "eval_save_per_experiment_images": bool(eval_save_per_experiment_images),
    }
    _write_json(os.path.join(out_dir, "eval_metadata.json"), meta)

    if run_mixture_eval:
        # -----------------------
        # Mixture evaluation (subset for speed)
        # -----------------------
        real_all = train_dataset.samples.detach().cpu().reshape(-1)
        n_total = int(real_all.shape[0])
        n_eval = int(min(max(1, eval_max_mixture_samples), n_total))

        rng = np.random.default_rng(int(seed))
        idx = rng.choice(n_total, size=n_eval, replace=False)
        real = real_all[idx].unsqueeze(1)

        samples_per_combo = int(train_dataset.samples_per_combo)
        combo_idx = torch.tensor(idx // samples_per_combo, dtype=torch.long)
        conds_raw = train_dataset.conds_raw.detach().cpu()[combo_idx]
        if normalize_conds:
            conds = normalize_conds_minus1_1(
                conds_raw,
                (ranges.m, ranges.mu, ranges.K, ranges.delta, ranges.omega),
            )
        else:
            conds = conds_raw

        gen = _generate_model_samples(
            model=model,
            conds=conds,
            latent_dim=int(model.latent_dim),
            batch_size=int(batch_size),
            seed=int(seed) + 10,
        )

        x = _to_1d_numpy(real)
        y = _to_1d_numpy(gen)

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(IEEE_DOUBLE_COL_IN, 2.6),
            dpi=300,
        )
        _empirical_qq(axes[0], x, y, title="Mixture QQ: real vs generated")
        _hist_kde(axes[1], x, y, title="Mixture KDE/Hist: real vs generated")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "mixture_qq_hist.png"), bbox_inches="tight")
        plt.close(fig)

        mix_metrics = _two_sample_metrics(x, y)
        _write_json(os.path.join(out_dir, "mixture_metrics.json"), mix_metrics)

    # -----------------------
    # Conditional evaluation (in/out range)
    # -----------------------
    mftr_gen = mftr(name="mftr", n_inverse_terms=2000, dist_type=str(dist_type))

    results: dict[str, Any] = {"in_range": [], "out_of_range": []}

    conds_in_raw = sample_params_within(ranges, int(eval_num_params_in), int(seed) + 100)
    conds_out_raw = sample_params_outside(
        ranges, int(eval_num_params_out), int(seed) + 200
    )

    def _compute_case(*, global_index: int, local_index: int, c: torch.Tensor) -> dict[str, Any]:
        m, mu, K, delta, omega = (float(v) for v in c.tolist())

        theo = mftr_gen.rvs(
            m,
            K,
            delta,
            mu,
            omega,
            size=int(eval_num_samples_per_param),
            random_state=int(seed) + 1000 + global_index,
        )
        theo_t = torch.tensor(theo, dtype=torch.float32).reshape(-1, 1)

        if normalize_conds:
            c_model = normalize_conds_minus1_1(
                c.reshape(1, 5),
                (ranges.m, ranges.mu, ranges.K, ranges.delta, ranges.omega),
            ).reshape(5)
        else:
            c_model = c

        cond_batch = c_model.reshape(1, 5).repeat(int(eval_num_samples_per_param), 1)
        gen_t = _generate_model_samples(
            model=model,
            conds=cond_batch,
            latent_dim=int(model.latent_dim),
            batch_size=int(batch_size),
            seed=int(seed) + 2000 + global_index,
        )

        tx = _to_1d_numpy(theo_t)
        ty = _to_1d_numpy(gen_t)

        return {
            "index": int(local_index),
            "global_index": int(global_index),
            "params": {
                "m": m,
                "mu": mu,
                "K": K,
                "delta": delta,
                "omega": omega,
            },
            "tx": tx,
            "ty": ty,
            "metrics": _two_sample_metrics(tx, ty),
        }

    in_cases: list[dict[str, Any]] = []
    out_cases: list[dict[str, Any]] = []

    global_index = 0
    for i in range(int(conds_in_raw.shape[0])):
        in_cases.append(
            _compute_case(global_index=global_index, local_index=i, c=conds_in_raw[i])
        )
        global_index += 1
    for i in range(int(conds_out_raw.shape[0])):
        out_cases.append(
            _compute_case(
                global_index=global_index, local_index=i, c=conds_out_raw[i]
            )
        )
        global_index += 1

    def _render_grid(*, label: str, cases: list[dict[str, Any]], out_path: str) -> None:
        n_rows = int(len(cases))
        if n_rows <= 0:
            return

        fig, axes = plt.subplots(
            n_rows,
            2,
            figsize=(IEEE_DOUBLE_COL_IN, 2.2 * n_rows),
            dpi=300,
            squeeze=False,
        )
        fig.suptitle(f"Conditional eval — {label}", y=0.995)
        fig.text(
            0.5,
            0.975,
            _format_ranges(ranges),
            ha="center",
            va="top",
        )

        for row, case in enumerate(cases):
            p = case["params"]
            title = (
                f"{label} #{case['index']:02d} — "
                + _format_params(p["m"], p["mu"], p["K"], p["delta"], p["omega"])
            )
            _empirical_qq(axes[row, 0], case["tx"], case["ty"], title=title)
            _generated_hist_kde_vs_theoretical_pdf(
                axes[row, 1],
                case["ty"],
                m=p["m"],
                mu=p["mu"],
                K=p["K"],
                delta=p["delta"],
                omega=p["omega"],
                title="Generated vs MFTR PDF",
                show_legend=(row == 0),
            )

        fig.tight_layout(rect=(0, 0, 1, 0.955))
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    def _maybe_save_per_experiment_images(
        *,
        label: str,
        cases: list[dict[str, Any]],
        subdir: str,
    ) -> None:
        if not eval_save_per_experiment_images:
            return
        if len(cases) <= 0:
            return

        img_dir = os.path.join(out_dir, subdir)
        _ensure_dir(img_dir)

        for case in cases:
            p = case["params"]

            fig, axes = plt.subplots(
                1,
                2,
                figsize=(IEEE_DOUBLE_COL_IN, 2.6),
                dpi=300,
            )
            fig.suptitle(
                f"{label} #{case['index']:02d} — "
                + _format_params(p["m"], p["mu"], p["K"], p["delta"], p["omega"]),
                y=0.995,
            )
            fig.text(
                0.5,
                0.965,
                _format_ranges(ranges),
                ha="center",
                va="top",
            )

            _empirical_qq(axes[0], case["tx"], case["ty"], title="QQ")
            _generated_hist_kde_vs_theoretical_pdf(
                axes[1],
                case["ty"],
                m=p["m"],
                mu=p["mu"],
                K=p["K"],
                delta=p["delta"],
                omega=p["omega"],
                title="Generated vs MFTR PDF",
                show_legend=True,
            )

            fig.tight_layout(rect=(0, 0, 1, 0.94))
            fname = f"conditional_{label}_{case['index']:02d}.png"
            fig.savefig(os.path.join(img_dir, fname), bbox_inches="tight")
            plt.close(fig)

            case["per_experiment_plot"] = os.path.join(subdir, fname)

    interpolation_grid = os.path.join(out_dir, "conditional_interpolation_grid.png")
    extrapolation_grid = os.path.join(out_dir, "conditional_extrapolation_grid.png")

    _render_grid(label="interpolation", cases=in_cases, out_path=interpolation_grid)
    _render_grid(label="extrapolation", cases=out_cases, out_path=extrapolation_grid)

    _maybe_save_per_experiment_images(
        label="interpolation",
        cases=in_cases,
        subdir="conditional_interpolation",
    )
    _maybe_save_per_experiment_images(
        label="extrapolation",
        cases=out_cases,
        subdir="conditional_extrapolation",
    )

    for case in in_cases:
        entry = {
            "index": int(case["index"]),
            "params": case["params"],
            "metrics": case["metrics"],
            "grid_plot": "conditional_interpolation_grid.png",
        }
        if eval_save_per_experiment_images and "per_experiment_plot" in case:
            entry["plot"] = case["per_experiment_plot"]
        else:
            entry["plot"] = "conditional_interpolation_grid.png"
        results["in_range"].append(entry)

    for case in out_cases:
        entry = {
            "index": int(case["index"]),
            "params": case["params"],
            "metrics": case["metrics"],
            "grid_plot": "conditional_extrapolation_grid.png",
        }
        if eval_save_per_experiment_images and "per_experiment_plot" in case:
            entry["plot"] = case["per_experiment_plot"]
        else:
            entry["plot"] = "conditional_extrapolation_grid.png"
        results["out_of_range"].append(entry)

    _write_json(os.path.join(out_dir, "conditional_metrics.json"), results)
