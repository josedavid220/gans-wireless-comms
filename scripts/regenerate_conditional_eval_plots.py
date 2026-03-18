from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from cgans import CGAN
from cgans.mftr_evaluation import MftrRanges, run_end_of_training_evaluation
from local_datasets.mftr_uniform_conditional_dataset import normalize_conds_minus1_1
from local_distributions import mftr


IEEE_DOUBLE_COL_IN = 7.16


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return start


def _apply_ieee_style(repo_root: Path) -> None:
    style = repo_root / "report" / "ieee.mplstyle"
    if style.exists():
        plt.style.use(str(style))


def _choose_three(entries: list[dict], *, mode: str) -> list[dict]:
    """Pick 3 representative entries.

    - interpolation: low/mid/high mu (diverse shapes)
    - extrapolation: best/median/worst by Wasserstein
    """
    if len(entries) <= 3:
        return entries

    if mode == "interpolation":
        mus = np.array([float(e["params"]["mu"]) for e in entries], dtype=float)
        lo_i = int(np.argmin(mus))
        hi_i = int(np.argmax(mus))
        mid_target = float((mus.min() + mus.max()) / 2.0)
        mid_i = int(np.argmin(np.abs(mus - mid_target)))
        picked = [entries[lo_i], entries[mid_i], entries[hi_i]]
        # Ensure uniqueness
        out: list[dict] = []
        seen = set()
        for e in picked:
            k = int(e.get("index", len(seen)))
            if k not in seen:
                out.append(e)
                seen.add(k)
        # Fill if any duplicates happened
        if len(out) < 3:
            for e in entries:
                k = int(e.get("index", len(seen)))
                if k not in seen:
                    out.append(e)
                    seen.add(k)
                if len(out) == 3:
                    break
        return out

    # extrapolation: enforce distinct mu values and prefer truly out-of-range mu.
    mus = np.array([float(e["params"]["mu"]) for e in entries], dtype=float)
    w = np.array([float(e["metrics"]["wasserstein"]) for e in entries], dtype=float)

    # Deduplicate by mu (rounded) to avoid repeated boundary values.
    unique_by_mu: dict[float, dict] = {}
    for e, mu_v in zip(entries, mus, strict=False):
        key = float(np.round(mu_v, 6))
        # Keep the worst (largest Wasserstein) for that mu.
        if key not in unique_by_mu or float(e["metrics"]["wasserstein"]) > float(unique_by_mu[key]["metrics"]["wasserstein"]):
            unique_by_mu[key] = e
    uniq = list(unique_by_mu.values())
    if len(uniq) <= 3:
        return uniq

    mus_u = np.array([float(e["params"]["mu"]) for e in uniq], dtype=float)
    w_u = np.array([float(e["metrics"]["wasserstein"]) for e in uniq], dtype=float)

    picked: list[dict] = []
    seen_mu: set[float] = set()

    def add_one(cands: list[dict], *, pick: str) -> None:
        nonlocal picked
        if not cands:
            return
        if pick == "worst":
            e = max(cands, key=lambda x: float(x["metrics"]["wasserstein"]))
        elif pick == "best":
            e = min(cands, key=lambda x: float(x["metrics"]["wasserstein"]))
        else:
            cands_sorted = sorted(cands, key=lambda x: float(x["metrics"]["wasserstein"]))
            e = cands_sorted[len(cands_sorted) // 2]
        mu_k = float(np.round(float(e["params"]["mu"]), 6))
        if mu_k not in seen_mu:
            picked.append(e)
            seen_mu.add(mu_k)

    # Prefer mu strictly above the training range (and within a sane cap).
    high = [e for e in uniq if 9.0 < float(e["params"]["mu"]) < 15.66]
    add_one(high, pick="median")

    # Fill remaining slots with best/median/worst by Wasserstein.
    order = np.argsort(w_u)
    candidates_ranked = [uniq[int(order[0])], uniq[int(order[len(order) // 2])], uniq[int(order[-1])]]
    for e in candidates_ranked:
        mu_k = float(np.round(float(e["params"]["mu"]), 6))
        if mu_k not in seen_mu:
            picked.append(e)
            seen_mu.add(mu_k)
        if len(picked) == 3:
            break

    if len(picked) < 3:
        for e in uniq:
            mu_k = float(np.round(float(e["params"]["mu"]), 6))
            if mu_k not in seen_mu:
                picked.append(e)
                seen_mu.add(mu_k)
            if len(picked) == 3:
                break

    return picked


def _gen_samples(
    *,
    model: CGAN,
    cond_raw: torch.Tensor,
    ranges: MftrRanges,
    normalize_conds: bool,
    n: int,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    # Prepare conditioning
    c = cond_raw.reshape(1, 5).to(dtype=torch.float32)
    if normalize_conds:
        c = normalize_conds_minus1_1(c, (ranges.m, ranges.mu, ranges.K, ranges.delta, ranges.omega))
    conds = c.repeat(int(n), 1)

    device = next(model.parameters()).device
    model = model.to(device)
    model.eval()

    out = []
    g = torch.Generator(device=device).manual_seed(int(seed))
    with torch.no_grad():
        for start in range(0, int(n), int(batch_size)):
            c_batch = conds[start : start + int(batch_size)].to(device)
            z = torch.randn(c_batch.shape[0], int(model.latent_dim), device=device, generator=g)
            x = model(z, c_batch)
            out.append(x.detach().cpu())
    y = torch.cat(out, dim=0).reshape(-1).numpy()
    return np.asarray(y, dtype=float)


def _theoretical_samples(
    *,
    mftr_gen,
    params: dict,
    n: int,
    seed: int,
) -> np.ndarray:
    m = float(params["m"])
    mu = float(params["mu"])
    K = float(params["K"])
    delta = float(params["delta"])
    omega = float(params["omega"])
    x = mftr_gen.rvs(m, K, delta, mu, omega, size=int(n), random_state=int(seed))
    return np.asarray(x, dtype=float).reshape(-1)


def _plot_case_row(ax_qq, ax_hist, *, x: np.ndarray, y: np.ndarray, mu: float, show_legend: bool) -> None:
    n = int(min(len(x), len(y)))
    xs = np.sort(x)[:n]
    ys = np.sort(y)[:n]
    ax_qq.scatter(xs, ys, s=4, alpha=0.35)
    lo = float(min(xs.min(), ys.min()))
    hi = float(max(xs.max(), ys.max()))
    ax_qq.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax_qq.set_title(f"$\\mu$={mu:.3f}")
    ax_qq.grid(True, alpha=0.2)

    sns.histplot(y, ax=ax_hist, stat="density", color="tab:blue", alpha=0.15, bins=60)
    sns.kdeplot(y, ax=ax_hist, label="Generated KDE", color="tab:blue", linewidth=2)
    sns.kdeplot(x, ax=ax_hist, label="MFTR (theoretical)", color="red", linewidth=2)
    if show_legend:
        ax_hist.legend()
    ax_hist.grid(True, alpha=0.2)



class _StubDataset:
    """Minimal dataset interface needed by run_end_of_training_evaluation.

    We only use it to provide conds_raw and metadata when regenerating plots.
    """

    def __init__(self, conds_raw: torch.Tensor, samples_per_combo: int):
        self.conds_raw = conds_raw
        self.samples_per_combo = int(samples_per_combo)
        self.combos = int(conds_raw.shape[0])
        # Not used when run_mixture_eval=False.
        self.samples = torch.empty((0, 1), dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate evaluation figures (IEEE sizing) for a trained cGAN version dir. "
            "This reruns evaluation; it does not retrain."
        )
    )
    parser.add_argument(
        "--version-dir",
        type=Path,
        required=True,
        help="Path like logs/mftr/cgan/version_16",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size used during evaluation sampling",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=3,
        help="Number of examples to keep for interpolation/extrapolation grids",
    )
    args = parser.parse_args()

    version_dir: Path = args.version_dir
    repo_root = _find_repo_root(version_dir)
    _apply_ieee_style(repo_root)
    ckpt_dir = version_dir / "checkpoints"
    test_results = version_dir / "test_results"

    if not ckpt_dir.exists():
        raise SystemExit(f"Missing checkpoints dir: {ckpt_dir}")
    if not test_results.exists():
        raise SystemExit(f"Missing test_results dir: {test_results}")

    # Pick a checkpoint (prefer best_model.ckpt if present, else first .ckpt)
    ckpt = ckpt_dir / "best_model.ckpt"
    if not ckpt.exists():
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        if not ckpts:
            raise SystemExit(f"No .ckpt files found under: {ckpt_dir}")
        ckpt = ckpts[0]

    meta_path = test_results / "eval_metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"Missing eval metadata: {meta_path}")
    meta = _load_json(meta_path)

    seed = int(meta.get("seed", 42))
    dist_type = str(meta.get("dist_type", "amplitude"))
    normalize_conds = bool(meta.get("normalize_conds", True))

    ranges_d = meta["ranges"]
    ranges = MftrRanges(
        m=tuple(ranges_d["m"]),
        mu=tuple(ranges_d["mu"]),
        K=tuple(ranges_d["K"]),
        delta=tuple(ranges_d["delta"]),
        omega=tuple(ranges_d["omega"]),
    )

    combos = int(meta.get("combos", 128))
    samples_per_combo = int(meta.get("samples_per_combo", 10000))

    # Determine how many conditional cases were evaluated originally.
    cond_metrics_path = test_results / "conditional_metrics.json"
    if not cond_metrics_path.exists():
        raise SystemExit(f"Missing conditional metrics: {cond_metrics_path}")
    cond_metrics = _load_json(cond_metrics_path)
    n_in = int(len(cond_metrics.get("in_range", [])))
    n_out = int(len(cond_metrics.get("out_of_range", [])))
    # Use the recorded sample size (n) when present.
    n_per = None
    if n_in > 0:
        n_per = int(cond_metrics["in_range"][0]["metrics"].get("n", 5000))
    elif n_out > 0:
        n_per = int(cond_metrics["out_of_range"][0]["metrics"].get("n", 5000))
    else:
        n_per = 5000

    # Load model
    model = CGAN.load_from_checkpoint(str(ckpt), map_location="cpu")
    model.eval()

    # Lightweight stub dataset (no MFTR sampling) for regenerating conditional plots.
    train_conds_path = test_results / "train_conds_raw.pt"
    if train_conds_path.exists():
        conds_raw = torch.load(train_conds_path, map_location="cpu")
        if not isinstance(conds_raw, torch.Tensor):
            raise SystemExit(f"Unexpected train_conds_raw.pt type: {type(conds_raw)}")
    else:
        # Fallback: create dummy conds to keep metadata consistent.
        conds_raw = torch.zeros((combos, 5), dtype=torch.float32)
    dataset = _StubDataset(conds_raw=conds_raw, samples_per_combo=samples_per_combo)

    # If we have metrics from the latest experiment, pick 3 representative cases
    # and regenerate cleaner, paper-friendly grids.
    in_entries = list(cond_metrics.get("in_range", []))
    out_entries = list(cond_metrics.get("out_of_range", []))

    # Auto-rerun evaluation if the recorded out-of-range cases are degenerate
    # (e.g., repeated mu at the training boundary), which makes the extrapolation
    # figure uninformative.
    if len(out_entries) > 0:
        mu_out = np.array([float(e["params"]["mu"]) for e in out_entries], dtype=float)
        uniq_mu = np.unique(np.round(mu_out, 6))
        if len(uniq_mu) < int(min(args.keep, len(out_entries))):
            run_end_of_training_evaluation(
                out_dir=str(test_results),
                model=model,
                train_dataset=dataset,
                ranges=ranges,
                dist_type=dist_type,
                seed=seed,
                eval_max_mixture_samples=int(50_000),
                eval_num_params_in=int(max(3, n_in if n_in > 0 else 3)),
                eval_num_params_out=int(max(5, n_out if n_out > 0 else 5)),
                eval_num_samples_per_param=int(n_per),
                normalize_conds=normalize_conds,
                batch_size=int(args.batch_size),
                eval_save_per_experiment_images=bool(meta.get("eval_save_per_experiment_images", False)),
                run_mixture_eval=False,
            )
            cond_metrics = _load_json(cond_metrics_path)
            in_entries = list(cond_metrics.get("in_range", []))
            out_entries = list(cond_metrics.get("out_of_range", []))

    keep = int(args.keep)
    in_sel = _choose_three(in_entries, mode="interpolation")[:keep]
    out_sel = _choose_three(out_entries, mode="extrapolation")[:keep]

    # Fall back to standard evaluation if metrics are missing.
    if len(in_sel) == 0 or len(out_sel) == 0:
        run_end_of_training_evaluation(
            out_dir=str(test_results),
            model=model,
            train_dataset=dataset,
            ranges=ranges,
            dist_type=dist_type,
            seed=seed,
            eval_max_mixture_samples=int(50_000),
            eval_num_params_in=int(min(keep, n_in if n_in > 0 else keep)),
            eval_num_params_out=int(min(keep, n_out if n_out > 0 else keep)),
            eval_num_samples_per_param=int(n_per),
            normalize_conds=normalize_conds,
            batch_size=int(args.batch_size),
            eval_save_per_experiment_images=bool(meta.get("eval_save_per_experiment_images", False)),
            run_mixture_eval=False,
        )
        print(f"Regenerated evaluation figures under: {test_results}")
        return

    mftr_gen = mftr(name="mftr", n_inverse_terms=2000, dist_type=str(dist_type))

    def render(selected: list[dict], *, label: str, out_png: Path, out_pdf: Path) -> None:
        n_rows = len(selected)
        fig, axes = plt.subplots(
            n_rows,
            2,
            figsize=(IEEE_DOUBLE_COL_IN, 2.0 * n_rows),
            dpi=300,
            squeeze=False,
        )
        fig.suptitle(f"Conditional eval — {label}", y=0.995)

        for row, entry in enumerate(selected):
            p = entry["params"]
            cond_raw = torch.tensor(
                [p["m"], p["mu"], p["K"], p["delta"], p["omega"]],
                dtype=torch.float32,
            )
            x = _theoretical_samples(mftr_gen=mftr_gen, params=p, n=n_per, seed=seed + 1000 + row)
            y = _gen_samples(
                model=model,
                cond_raw=cond_raw,
                ranges=ranges,
                normalize_conds=normalize_conds,
                n=n_per,
                seed=seed + 2000 + row,
                batch_size=int(args.batch_size),
            )

            _plot_case_row(
                axes[row, 0],
                axes[row, 1],
                x=x,
                y=y,
                mu=float(p["mu"]),
                show_legend=(row == 0),
            )

            if row != n_rows - 1:
                axes[row, 0].set_xlabel("")
                axes[row, 1].set_xlabel("")
            if row != (n_rows // 2):
                axes[row, 0].set_ylabel("")

        # Axis labels only once (bottom row)
        axes[-1, 0].set_xlabel("Real quantiles")
        axes[-1, 0].set_ylabel("Generated quantiles")

        fig.tight_layout(rect=(0, 0, 1, 0.955))
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

    # Overwrite the cluttered grids in the version folder and export PDFs to report/images.
    out_interp_png = test_results / "conditional_interpolation_grid.png"
    out_extra_png = test_results / "conditional_extrapolation_grid.png"
    out_interp_pdf = repo_root / "report" / "images" / "cgan_v16_interpolation.pdf"
    out_extra_pdf = repo_root / "report" / "images" / "cgan_v16_extrapolation.pdf"

    render(in_sel, label="interpolation (3 examples)", out_png=out_interp_png, out_pdf=out_interp_pdf)
    render(out_sel, label="extrapolation (3 examples)", out_png=out_extra_png, out_pdf=out_extra_pdf)

    print(f"Wrote: {out_interp_png}")
    print(f"Wrote: {out_extra_png}")
    print(f"Wrote: {out_interp_pdf}")
    print(f"Wrote: {out_extra_pdf}")


if __name__ == "__main__":
    main()
