from gan import GAN
import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
from config import get_args
from scipy import stats
import json
from datetime import datetime
import os
import os.path as path

args = get_args()

NUM_SAMPLES = args.num_samples
SCALE = args.scale
CKPT_PATH = args.ckpt_path
TESTS_SAVE_PATH = args.tests_save_path
PRECISION = 8
DEVICE = "cpu"

model = GAN.load_from_checkpoint(CKPT_PATH).to(DEVICE)
rng = np.random.default_rng(0)


def compute_gof_tests(samples, dist=stats.rayleigh, statistic="ks", n_mc_samples=1000):
    res = stats.goodness_of_fit(
        dist=dist,
        data=samples,
        statistic=statistic,
        known_params={"loc": 0},
        n_mc_samples=n_mc_samples,
        rng=rng,
    )

    return res


tests = ["ks", "ad"]
results = {
    "model_ckpt": path.abspath(CKPT_PATH),
    "generated_samples": NUM_SAMPLES,
    "theoretical_scale": SCALE,
    "tests": dict()
}

with torch.no_grad():
    z = torch.randn(NUM_SAMPLES, model.latent_dim).to(DEVICE)
    samples = model(z, SCALE).squeeze().to(DEVICE)

print("Starting tests...")
for test in tests:
    result = compute_gof_tests(samples)
    results["tests"][test] = {
        "statistic": round(result.statistic, PRECISION),
        "pvalue": round(result.pvalue, PRECISION),
        "estimated-scale": round(result.fit_result.params.scale, PRECISION),
    }

print(results)

if not path.exists(TESTS_SAVE_PATH):
    os.makedirs(TESTS_SAVE_PATH, exist_ok=True)

date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = path.join(TESTS_SAVE_PATH, date)
os.makedirs(results_dir, exist_ok=True)
file_save_path = path.join(results_dir, "gof_tests.json")

with open(file_save_path, "w") as file:
    json.dump(results, file, indent=4)

print(f"Saved tests results to {file_save_path}")

print("Generating plots...")

def generate_distribution_subplots(num_samples, scales):
    linewidth = 1.5
    with torch.no_grad():
        fig, axes = plt.subplots(
            len(scales) // 2,
            2,
            figsize=(20, 5 * len(scales) // 2),
            sharey=True,
            # sharex=True,
        )
        if len(scales) == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, scale in enumerate(scales):
            z = torch.randn(num_samples, MODEL.latent_dim)
            generated_samples = MODEL(z, scale=scale)
            x = np.linspace(0, max(generated_samples), 100)
            pdf = rayleigh.pdf(x, scale=scale)
            axes[i].plot(x, pdf, "#1f78b4", linewidth=linewidth)
            # axes[i].set_label(rf"Theoretical Rayleigh PDF ($\sigma = {scale:.2f}$)")
            sns.histplot(
                generated_samples.numpy().flatten(),
                stat="density",
                kde=False,
                # label="Sample Distribution",
                ax=axes[i],
                color="orange",
                alpha=0.1,
            )
            sns.kdeplot(
                generated_samples.numpy().flatten(),
                color="#e31a1c",
                # label='KDE estimation',
                ax=axes[i],
                linewidth=linewidth,
                # fill=True,
            )
            # axes[i].legend()
            axes[i].set_title(rf"$\sigma = {scale:.2f}$")
            axes[i].set_xlim([0, 10])

        fig.suptitle("Rayleigh Distribution Approximation by GAN", fontsize=16)
        fig.legend(
            labels=[
                "Theoretical Rayleigh PDF",
                "KDE estimation",
                "Sample Distribution",
            ],
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.95),
            fontsize=12,
        )
        plt.show()


# generate_distribution_subplots(
#     # NUM_SAMPLES, scales=[np.random.random() * 10 for _ in range(6)]
#     NUM_SAMPLES,
#     scales=[i for i in range(1, 5)],
# )
