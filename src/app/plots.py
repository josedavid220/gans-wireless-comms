from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns

from local_distributions.mftr_dist import pdf_mftr


def _apply_style() -> None:
    try:
        plt.style.use("default")
    except Exception:
        pass


def make_comparison_figure(
    *,
    real: np.ndarray,
    generated: np.ndarray,
    m: float,
    mu: float,
    K: float,
    delta: float,
    omega: float,
) -> Figure:
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), dpi=140)

    # QQ plot
    n = min(len(real), len(generated))
    xr = np.sort(real)[:n]
    yg = np.sort(generated)[:n]
    ax = axes[0]
    ax.scatter(xr, yg, s=7, alpha=0.35)
    lo = float(min(xr.min(), yg.min()))
    hi = float(max(xr.max(), yg.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_title("QQ plot")
    ax.set_xlabel("Real quantiles")
    ax.set_ylabel("Generated quantiles")
    ax.grid(alpha=0.2)

    # Hist + KDE + theoretical MFTR PDF
    ax = axes[1]
    sns.histplot(generated, ax=ax, stat="density", color="#1f77b4", alpha=0.15, bins=60)
    sns.kdeplot(generated, ax=ax, color="#1f77b4", linewidth=2, label="Generated KDE")

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
    ax.plot(x_range, theoretical_pdf, color="red", linewidth=2, label="MFTR PDF")
    ax.set_title("Distribution comparison")
    ax.set_xlabel("Envelope")
    ax.set_ylabel("Density")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    fig.suptitle(
        f"m={m:.3f}, mu={mu:.3f}, K={K:.3f}, delta={delta:.3f}, omega={omega:.3f}",
        y=1.01,
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def _draw_box(ax, x, y, w, h, text, fc="#f5f7fa", ec="#2f4f4f"):
    rect = Rectangle((x, y), w, h, linewidth=1.4, edgecolor=ec, facecolor=fc)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def make_pipeline_diagram() -> Figure:
    fig, ax = plt.subplots(figsize=(8.4, 2.2), dpi=140)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _draw_box(ax, 0.02, 0.25, 0.20, 0.5, "Uniform MFTR\nparameter combos\n[m, mu, K, delta, omega]")
    _draw_box(ax, 0.28, 0.25, 0.18, 0.5, "MFTR sampler\n(theoretical)")
    _draw_box(ax, 0.52, 0.25, 0.20, 0.5, "cGAN training\n(BCE + label smoothing\nD updates > G updates)")
    _draw_box(ax, 0.78, 0.25, 0.20, 0.5, "Evaluation\nQQ + hist/KDE + PDF\nKS/CvM/Wass/MAE/MSE")

    for x0, x1 in [(0.22, 0.28), (0.46, 0.52), (0.72, 0.78)]:
        ax.annotate("", xy=(x1, 0.50), xytext=(x0, 0.50), arrowprops=dict(arrowstyle="->", lw=1.6))

    fig.tight_layout()
    return fig


def make_architecture_diagram() -> Figure:
    fig, ax = plt.subplots(figsize=(8.4, 3.1), dpi=140)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.20, 0.93, "Generator", fontsize=11, weight="bold")
    ax.text(0.70, 0.93, "Discriminator", fontsize=11, weight="bold")

    _draw_box(ax, 0.04, 0.62, 0.17, 0.20, "z ~ N(0, I)\n(latent)", fc="#fff7e6")
    _draw_box(ax, 0.04, 0.30, 0.17, 0.20, "Condition\n[m, mu, K, delta, omega]", fc="#eaf4ff")
    _draw_box(ax, 0.27, 0.30, 0.17, 0.20, "Cond embedding\nMLP")
    _draw_box(ax, 0.27, 0.62, 0.17, 0.20, "Concat [z, emb]")
    _draw_box(ax, 0.50, 0.62, 0.17, 0.20, "Generator MLP\n512-256-128-1\nSoftplus out")

    _draw_box(ax, 0.50, 0.18, 0.17, 0.20, "Real/Fake sample\n(1D envelope)", fc="#fff7e6")
    _draw_box(ax, 0.72, 0.30, 0.17, 0.20, "Cond embedding\nMLP")
    _draw_box(ax, 0.72, 0.62, 0.17, 0.20, "Concat [x, emb]")
    _draw_box(ax, 0.90, 0.62, 0.09, 0.20, "D MLP\n512-256-128-1")

    arrows = [
        ((0.21, 0.72), (0.27, 0.72)),
        ((0.21, 0.40), (0.27, 0.40)),
        ((0.44, 0.72), (0.50, 0.72)),
        ((0.67, 0.72), (0.72, 0.72)),
        ((0.67, 0.28), (0.72, 0.40)),
        ((0.89, 0.72), (0.90, 0.72)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.text(0.50, 0.49, "Generated sample feeds D as fake sample", ha="center", fontsize=9)
    fig.tight_layout()
    return fig
