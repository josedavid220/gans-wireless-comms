from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


IEEE_COL_IN = 3.5


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return start


def apply_ieee_style(repo_root: Path) -> None:
    style_path = repo_root / "report" / "ieee.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))


def read_scalars(acc: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
    ev = acc.Scalars(tag)
    steps = np.array([e.step for e in ev], dtype=int)
    vals = np.array([e.value for e in ev], dtype=float)
    return steps, vals


def main() -> None:
    repo_root = find_repo_root(Path(__file__))
    apply_ieee_style(repo_root)

    version_dir = repo_root / "logs" / "mftr" / "cgan" / "version_16"
    event_files = sorted(version_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise SystemExit(f"No TensorBoard event files found in: {version_dir}")

    acc = EventAccumulator(str(event_files[-1]), size_guidance={"scalars": 0})
    acc.Reload()
    tags = set(acc.Tags().get("scalars", []))

    required = ["d_loss", "g_loss", "train_real_prob", "train_fake_prob", "real_prob", "fake_prob"]
    missing = [t for t in required if t not in tags]
    if missing:
        raise SystemExit(f"Missing scalar tags in events: {missing}\nAvailable: {sorted(tags)}")

    # x-axis as epoch index (one point per epoch)
    _, d_loss = read_scalars(acc, "d_loss")
    _, g_loss = read_scalars(acc, "g_loss")
    epochs = np.arange(1, len(d_loss) + 1)

    out_dir = repo_root / "report" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------
    # Loss curves
    # -----------------
    fig, ax = plt.subplots(figsize=(IEEE_COL_IN, 2.2), dpi=300)
    ax.plot(epochs, d_loss, label="D loss", linewidth=1.6)
    ax.plot(epochs, g_loss, label="G loss", linewidth=1.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training losses")
    ax.legend(loc="best")
    fig.tight_layout(pad=0.2)

    out_losses = out_dir / "cgan_v16_training_losses.pdf"
    fig.savefig(out_losses, bbox_inches="tight")
    plt.close(fig)

    # -----------------
    # Probability curves
    # -----------------
    _, tr_real = read_scalars(acc, "train_real_prob")
    _, tr_fake = read_scalars(acc, "train_fake_prob")
    _, va_real = read_scalars(acc, "real_prob")
    _, va_fake = read_scalars(acc, "fake_prob")

    fig, ax = plt.subplots(figsize=(IEEE_COL_IN, 2.2), dpi=300)
    ax.plot(epochs, tr_real, label="Train real", linewidth=1.6)
    ax.plot(epochs, tr_fake, label="Train fake", linewidth=1.6)
    ax.plot(epochs, va_real, label="Val real", linewidth=1.6, linestyle="--")
    ax.plot(epochs, va_fake, label="Val fake", linewidth=1.6, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean discriminator probability")
    ax.set_ylim(0.35, 0.55)
    ax.set_title("Discriminator probabilities")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout(pad=0.2)

    out_probs = out_dir / "cgan_v16_training_probabilities.pdf"
    fig.savefig(out_probs, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {out_losses}")
    print(f"Wrote: {out_probs}")


if __name__ == "__main__":
    main()
