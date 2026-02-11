import math
import os

import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from local_distributions.mftr_dist import pdf_mftr


def save_mftr_conditional_comparison_figure(
    trainer,
    model,
    conds_raw: torch.Tensor,
    num_samples: int,
    max_cols: int = 2,
    filename: str = "conditional_comparison.png",
):
    if not trainer.logger or not trainer.logger.log_dir:
        return

    num_combos = int(conds_raw.shape[0])
    cols = min(int(max_cols), num_combos)
    rows = int(math.ceil(num_combos / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows), dpi=300)
    axes = np.atleast_1d(axes).reshape(-1)

    model.eval()
    device = model.device

    with torch.no_grad():
        for i in range(num_combos):
            m, mu, K, delta, omega = conds_raw[i].cpu().tolist()
            params_str = (
                f"m={int(m)}, mu={int(mu)}, K={float(K):.2f}, "
                f"δ={float(delta):.2f}, ω={float(omega):.2f}"
            )

            cond = conds_raw[i].to(device).unsqueeze(0).repeat(num_samples, 1)
            z = torch.randn(
                num_samples,
                model.latent_dim,
                device=device,
                generator=torch.Generator(device=device).manual_seed(42 + i),
            )
            samples = model(z, cond).squeeze().detach().cpu().numpy()

            ax = axes[i]
            sns.histplot(
                samples,
                alpha=0.6,
                color="skyblue",
                label="Generated",
                stat="density",
                ax=ax,
            )
            sns.kdeplot(samples, color="darkblue", linewidth=2, label="KDE", ax=ax)

            x_range = np.linspace(1e-6, 3.0 * np.sqrt(float(omega)), 1000)
            theoretical_pdf = pdf_mftr(
                x_range**2,
                m=int(m),
                K=float(K),
                delta=float(delta),
                mu=int(mu),
                omega=float(omega),
            )
            theoretical_pdf *= 2.0 * x_range

            ax.plot(
                x_range,
                theoretical_pdf,
                "r-",
                linewidth=2,
                label="MFTR PDF",
            )

            ax.set_title(params_str, fontsize=10, fontweight="bold")
            ax.set_xlabel("Value", fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    for j in range(num_combos, len(axes)):
        axes[j].axis("off")

    sns.despine()
    fig.tight_layout()

    out_dir = os.path.join(trainer.logger.log_dir, "test_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    trainer.logger.experiment.add_figure(  # type: ignore
        "mftr_conditional_comparison",
        fig,
        close=False,
    )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved comparison plot to: {os.path.abspath(out_path)}")
