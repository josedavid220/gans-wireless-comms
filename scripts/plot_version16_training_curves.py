from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return start


def _apply_ieee_style(repo_root: Path) -> None:
    style_path = repo_root / "report" / "ieee.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))


def _read_scalar(acc: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
    events = acc.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=int)
    vals = np.array([e.value for e in events], dtype=float)
    return steps, vals


def _to_epoch_index(steps: np.ndarray) -> np.ndarray:
    # In this project, these scalars are logged once per epoch.
    return np.arange(1, len(steps) + 1, dtype=int)


def _auto_prob_ylim(
    *series: np.ndarray,
    pad_frac: float = 0.20,
    min_span: float = 0.06,
) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(s, dtype=float).reshape(-1) for s in series])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0

    # Robust range to avoid a single spike dominating the axis.
    lo = float(np.quantile(vals, 0.01))
    hi = float(np.quantile(vals, 0.99))
    if not (lo < hi):
        mid = float(np.median(vals))
        lo, hi = mid - min_span / 2.0, mid + min_span / 2.0

    span = max(hi - lo, min_span)
    pad = span * float(pad_frac)
    y0 = max(0.0, lo - pad)
    y1 = min(1.0, hi + pad)

    # Ensure a minimum visible span even after clipping.
    if y1 - y0 < min_span:
        mid = float(np.clip((y0 + y1) / 2.0, 0.0, 1.0))
        y0 = max(0.0, mid - min_span / 2.0)
        y1 = min(1.0, mid + min_span / 2.0)
    return float(y0), float(y1)


def main() -> None:
    repo_root = _find_repo_root(Path(__file__))
    _apply_ieee_style(repo_root)

    version_dir = repo_root / "logs" / "mftr" / "cgan" / "version_16"
    event_files = sorted(version_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise SystemExit(f"No TensorBoard event files found in: {version_dir}")

    event_path = event_files[-1]

    acc = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    acc.Reload()

    tags = set(acc.Tags().get("scalars", []))

    required = ["d_loss", "g_loss", "train_real_prob", "train_fake_prob", "real_prob", "fake_prob"]
    missing = [t for t in required if t not in tags]
    if missing:
        raise SystemExit(f"Missing required scalar tags: {missing}. Available: {sorted(tags)}")

    # --- Loss curves ---
    s_d, d_loss = _read_scalar(acc, "d_loss")
    s_g, g_loss = _read_scalar(acc, "g_loss")
    e_d = _to_epoch_index(s_d)
    e_g = _to_epoch_index(s_g)

    fig, (ax_d, ax_g) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(3.5, 2.2),
        dpi=300,
        gridspec_kw={"height_ratios": [1, 1]},
    )
    ax_d.plot(e_d, d_loss, label="Discriminator loss")
    ax_d.set_ylabel("D loss")
    ax_d.legend(loc="best")

    ax_g.plot(e_g, g_loss, label="Generator loss", color="tab:orange")
    ax_g.set_ylabel("G loss")
    ax_g.set_xlabel("Epoch")
    ax_g.legend(loc="best")

    fig.tight_layout(pad=0.2, h_pad=0.15)

    out_loss = repo_root / "report" / "images" / "cgan_v16_training_losses.pdf"
    out_loss.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_loss, bbox_inches="tight")
    plt.close(fig)

    # --- Probability curves ---
    s_tr, tr_real = _read_scalar(acc, "train_real_prob")
    s_tf, tr_fake = _read_scalar(acc, "train_fake_prob")
    s_vr, v_real = _read_scalar(acc, "real_prob")
    s_vf, v_fake = _read_scalar(acc, "fake_prob")

    e = _to_epoch_index(s_tr)

    fig, ax = plt.subplots(figsize=(3.5, 2.2), dpi=300)
    ax.plot(e, tr_real, label="Train real prob")
    ax.plot(e, tr_fake, label="Train fake prob")
    ax.plot(e, v_real, label="Val real prob", linestyle="--")
    ax.plot(e, v_fake, label="Val fake prob", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Discriminator probability")
    y0, y1 = _auto_prob_ylim(tr_real, tr_fake, v_real, v_fake)
    ax.set_ylim(y0, y1)
    ax.legend(loc="best")
    fig.tight_layout(pad=0.2)

    out_prob = repo_root / "report" / "images" / "cgan_v16_training_probs.pdf"
    out_prob.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prob, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {out_loss}")
    print(f"Wrote: {out_prob}")


if __name__ == "__main__":
    main()
