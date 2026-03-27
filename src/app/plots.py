from __future__ import annotations

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns

from local_distributions.mftr_dist import pdf_mftr


PALETTE = {
    "ink": "#1f2937",
    "grid": "#d1d5db",
    "gen": "#0b84a5",
    "real": "#c1121f",
    "hist": "#9aa5b1",
    "accent": "#d97706",
}


def _apply_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        rc={
            "axes.edgecolor": PALETTE["ink"],
            "axes.linewidth": 1.1,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "font.size": 11,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
        },
    )


def make_qq_figure(
    *,
    real: np.ndarray,
    generated: np.ndarray,
) -> Figure:
    _apply_style()
    fig = Figure(figsize=(5.6, 4.4), dpi=150)
    ax = fig.add_subplot(1, 1, 1)

    n = min(len(real), len(generated))
    xr = np.sort(real)[:n]
    yg = np.sort(generated)[:n]
    ax.scatter(xr, yg, s=9, alpha=0.22, color=PALETTE["gen"], linewidths=0)
    lo = float(min(xr.min(), yg.min()))
    hi = float(max(xr.max(), yg.max()))
    ax.plot([lo, hi], [lo, hi], color=PALETTE["accent"], linestyle="--", linewidth=1.6)
    ax.set_xlabel("Real quantiles")
    ax.set_ylabel("Generated quantiles")
    ax.grid(alpha=0.35)
    fig.tight_layout()
    return fig


def make_density_figure(
    *,
    generated: np.ndarray,
    m: float,
    mu: float,
    K: float,
    delta: float,
    omega: float,
) -> Figure:
    _apply_style()
    fig = Figure(figsize=(5.9, 4.4), dpi=150)
    ax = fig.add_subplot(1, 1, 1)

    sns.histplot(
        generated,
        ax=ax,
        stat="density",
        color=PALETTE["hist"],
        alpha=0.28,
        bins=70,
        edgecolor="#5b6572",
        linewidth=0.55,
    )
    sns.kdeplot(generated, ax=ax, color=PALETTE["gen"], linewidth=2.4, label="Generated KDE")

    x_range = np.linspace(1e-6, 3.2 * np.sqrt(float(omega)), 1200)
    theoretical_pdf = pdf_mftr(
        x_range**2,
        m=int(m),
        K=float(K),
        delta=float(delta),
        mu=int(mu),
        omega=float(omega),
    )
    theoretical_pdf = theoretical_pdf * (2.0 * x_range)
    ax.plot(x_range, theoretical_pdf, color=PALETTE["real"], linewidth=2.5, label="MFTR PDF")
    ax.set_xlabel("Envelope")
    ax.set_ylabel("Density")
    ax.legend(loc="best")
    ax.grid(alpha=0.35)
    fig.tight_layout()
    return fig


def _ecdf(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(np.asarray(a, dtype=float).reshape(-1))
    ys = np.arange(1, xs.size + 1, dtype=float) / xs.size
    return xs, ys


def make_cdf_figure(*, real: np.ndarray, generated: np.ndarray) -> Figure:
    _apply_style()
    fig = Figure(figsize=(5.9, 4.4), dpi=150)
    ax = fig.add_subplot(1, 1, 1)

    xr, fr = _ecdf(real)
    xg, fg = _ecdf(generated)

    ax.step(xr, fr, where="post", color=PALETTE["real"], linewidth=2.2, label="Real empirical CDF")
    ax.step(xg, fg, where="post", color=PALETTE["gen"], linewidth=2.2, label="Generated empirical CDF")
    ax.set_xlabel("Envelope")
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.35)
    fig.tight_layout()
    return fig

def _draw_box(ax, x, y, w, h, text, fc="#f7fafc", ec="#334155", lw: float = 1.5):
    rect = Rectangle((x, y), w, h, linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, color=PALETTE["ink"])


def make_pipeline_diagram() -> Figure:
    fig = Figure(figsize=(9.8, 2.9), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    _draw_box(ax, 0.02, 0.20, 0.22, 0.60, "1) Uniform parameter\ncombos\n[m, mu, K, delta, omega]", fc="#eef6ff")
    _draw_box(ax, 0.28, 0.20, 0.19, 0.60, "2) MFTR reference\nsampling", fc="#fff7ed")
    _draw_box(ax, 0.51, 0.20, 0.22, 0.60, "3) cGAN training\n(BCE + smoothing\nD updates > G updates)", fc="#f2fdf5")
    _draw_box(ax, 0.77, 0.20, 0.21, 0.60, "4) Interactive eval\nQQ / PDF / CDF\nKS, CvM, AD, JS, H", fc="#f7f4ff")

    for x0, x1 in [(0.24, 0.28), (0.47, 0.51), (0.73, 0.77)]:
        ax.annotate(
            "",
            xy=(x1, 0.50),
            xytext=(x0, 0.50),
            arrowprops=dict(arrowstyle="->", lw=2.0, color="#111827"),
        )

    ax.text(0.02, 0.90, "Pipeline", fontsize=13, weight="bold", color=PALETTE["ink"])

    fig.tight_layout()
    return fig


def make_architecture_diagram() -> Figure:
    fig = Figure(figsize=(9.8, 4.6), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.15, 0.94, "Generator branch", fontsize=13, weight="bold", color="#0f766e")
    ax.text(0.63, 0.94, "Discriminator branch", fontsize=13, weight="bold", color="#7c2d12")

    _draw_box(ax, 0.03, 0.66, 0.18, 0.18, "Latent input\nz ~ N(0, I)", fc="#fff7ed", ec="#9a3412")
    _draw_box(ax, 0.03, 0.36, 0.18, 0.18, "Condition\n[m, mu, K, delta, omega]", fc="#eff6ff", ec="#1d4ed8")
    _draw_box(ax, 0.25, 0.36, 0.18, 0.18, "Condition\nembedding MLP", fc="#f8fafc")
    _draw_box(ax, 0.25, 0.66, 0.18, 0.18, "Concat\n[z, emb]", fc="#f8fafc")
    _draw_box(ax, 0.47, 0.66, 0.20, 0.18, "Generator MLP\n512 -> 256 -> 128 -> 1\nSoftplus output", fc="#ecfeff", ec="#0f766e")

    _draw_box(ax, 0.47, 0.12, 0.20, 0.18, "Sample x\n(real or generated)", fc="#fff7ed", ec="#9a3412")
    _draw_box(ax, 0.71, 0.36, 0.18, 0.18, "Condition\nembedding MLP", fc="#f8fafc")
    _draw_box(ax, 0.71, 0.66, 0.18, 0.18, "Concat\n[x, emb]", fc="#f8fafc")
    _draw_box(ax, 0.91, 0.66, 0.08, 0.18, "Discriminator\nMLP\n512->256->128->1", fc="#fef2f2", ec="#7f1d1d")

    arrows = [
        ((0.21, 0.75), (0.25, 0.75)),
        ((0.21, 0.45), (0.25, 0.45)),
        ((0.43, 0.75), (0.47, 0.75)),
        ((0.67, 0.75), (0.71, 0.75)),
        ((0.67, 0.21), (0.71, 0.45)),
        ((0.89, 0.75), (0.91, 0.75)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=2.0, color="#111827"),
        )

    ax.text(0.50, 0.58, "Generated sample is routed to D as fake input", ha="center", fontsize=10, color=PALETTE["ink"])
    fig.tight_layout()
    return fig
