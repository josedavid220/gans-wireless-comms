from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils.ieee_plot_style import apply_ieee_style, find_repo_root
from local_distributions.mftr_dist import pdf_mftr


def mftr_amplitude_pdf(
    r: np.ndarray,
    *,
    m: float,
    mu: float,
    K: float,
    delta: float,
    omega: float,
    n_inverse_terms: int = 4000,
) -> np.ndarray:
    """Amplitude PDF f_R(r) derived from MFTR power PDF f_G(g).

    If G = R^2, then f_R(r) = f_G(r^2) * 2r.
    """
    r = np.asarray(r, dtype=float)
    g = r**2
    f_g = pdf_mftr(
        g,
        m=float(m),
        K=float(K),
        delta=float(delta),
        mu=float(mu),
        omega=float(omega),
        n_inverse_terms=int(n_inverse_terms),
    )
    return f_g * (2.0 * r)


def main() -> None:
    repo_root = find_repo_root()

    parser = argparse.ArgumentParser(
        description="Generate MFTR amplitude PDF figure varying mu for two cases."
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=repo_root / "report" / "images" / "mftr_mu_cases.pdf",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=repo_root / "report" / "images" / "mftr_mu_cases.png",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    apply_ieee_style(repo_root)

    # Cases from the reference figure caption
    case_a = {"delta": 0.9, "m": 8.0, "K": 8.0, "omega": 2.0}
    case_b = {"delta": 0.9, "m": 4.0, "K": 15.0, "omega": 1.0}

    mus = [1, 3, 5, 7]
    mu_styles = {
        1: {"color": "tab:orange", "linestyle": ":"},
        3: {"color": "tab:blue", "linestyle": "--"},
        5: {"color": "tab:red", "linestyle": "-."},
        7: {"color": "black", "linestyle": "-"},
    }

    r = np.linspace(1e-6, 2.5, 2000)

    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    for mu in mus:
        style = mu_styles[int(mu)]

        y_a = mftr_amplitude_pdf(r, mu=mu, **case_a)
        y_b = mftr_amplitude_pdf(r, mu=mu, **case_b)

        ax.plot(
            r,
            y_a,
            label=rf"$\mu$ = {int(mu)}",
            **style,
        )
        ax.plot(
            r,
            y_b,
            label=None,
            **style,
        )

    ax.set_xlim(0.0, 2.5)
    ax.set_ylim(0.0, 1.2)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$f_R(r)$")

    # Annotations (match the reference figure intent)
    # Pick anchor points on the mu=7 curves.
    y_a_mu7 = mftr_amplitude_pdf(r, mu=7, **case_a)
    y_b_mu7 = mftr_amplitude_pdf(r, mu=7, **case_b)

    x_a = 1.90
    y_a = float(np.interp(x_a, r, y_a_mu7))
    ax.annotate(
        "Case A",
        xy=(x_a, y_a),
        xytext=(2.05, 0.72),
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )

    x_b = 1.10
    y_b = float(np.interp(x_b, r, y_b_mu7))
    ax.annotate(
        "Case B",
        xy=(x_b, y_b),
        xytext=(1.25, 0.90),
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )

    ax.legend(loc="upper right")
    fig.tight_layout(pad=0.2)

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf)
    fig.savefig(args.out_png, dpi=int(args.dpi))
    plt.close(fig)

    print(f"Wrote: {args.out_pdf}")
    print(f"Wrote: {args.out_png}")


if __name__ == "__main__":
    main()
