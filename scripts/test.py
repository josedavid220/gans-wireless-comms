from gans import GAN, WGAN_GP
import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
from config import get_args
from scipy import stats
import json
import os
import os.path as path

args = get_args()

NUM_SAMPLES = args.num_samples
TESTS_SAVE_PATH = args.tests_save_path
PRECISION = 8
DEVICE = "cpu"
GAN_TYPE = args.gan_type
VERSION = args.version
DATASET = args.dataset

version_dir = path.join("..", "logs", DATASET, GAN_TYPE, f"version_{VERSION}")
ckpt_dir = path.join(version_dir, "checkpoints")
theoretical_params_path = path.join(version_dir, "distribution_params.json")

ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
if not ckpt_files:
    raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")

CKPT_PATH = path.join(ckpt_dir, ckpt_files[0])


def get_model():
    match GAN_TYPE:
        case "gan":
            model = GAN
        case "wgan_gp":
            model = WGAN_GP
        case _:
            raise ValueError(f"Unsupported GAN type: {GAN_TYPE}")

    return model.load_from_checkpoint(CKPT_PATH).to(DEVICE)


model = get_model()


def compute_gof_tests(samples, dist_name="rayleigh", statistic="ks", n_mc_samples=1000):
    match dist_name:
        case "rayleigh":
            dist = stats.rayleigh
        case "nakagami":
            dist = stats.nakagami
        case _:
            raise ValueError(f"Unsupported distribution: {dist_name}")

    rng = np.random.default_rng(0)
    res = stats.goodness_of_fit(
        dist=dist,
        data=samples,
        statistic=statistic,
        n_mc_samples=n_mc_samples,
        rng=rng,
    )

    return res


with open(theoretical_params_path, "r") as f:
    theoretical_params = json.load(f)

tests = ["ks", "ad"]
results = {
    "model_ckpt": path.abspath(CKPT_PATH),
    "dataset": DATASET,
    "gan_type": GAN_TYPE,
    "generated_samples": NUM_SAMPLES,
    "theoretical_params_path": theoretical_params,
    "tests": dict(),
}

with torch.no_grad():
    z = torch.randn(NUM_SAMPLES, model.latent_dim).to(DEVICE)
    samples = model(z).squeeze().to(DEVICE)

print("Starting tests...")
for test in tests:
    result = compute_gof_tests(samples, dist_name=DATASET, statistic=test)
    results["tests"][test] = {
        "statistic": round(result.statistic, PRECISION),
        "pvalue": round(result.pvalue, PRECISION),
        "estimated-params": result.fit_result.params._asdict(),
    }

print(results)

if not path.exists(TESTS_SAVE_PATH):
    os.makedirs(TESTS_SAVE_PATH, exist_ok=True)

# date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = path.join(TESTS_SAVE_PATH, DATASET, GAN_TYPE, f"version_{VERSION}")
os.makedirs(results_dir, exist_ok=True)
file_save_path = path.join(results_dir, "gof_tests.json")

with open(file_save_path, "w") as file:
    json.dump(results, file, indent=4)

print(f"Saved tests results to {file_save_path}")

print("Generating plots...")


def plot_distribution_comparison(samples, dist_name, theoretical_params, save_path):
    """Generate a comparison plot showing histogram, KDE, and theoretical distribution."""
    plt.figure(figsize=(10, 6))

    # Convert samples to numpy for plotting
    samples_np = samples

    # Create histogram with density=True
    sns.histplot(
        samples_np,
        alpha=0.6,
        color="skyblue",
        label="Generated Samples",
        stat="density",
    )

    # Add KDE plot
    sns.kdeplot(samples_np, color="darkblue", linewidth=2, label="KDE")

    # Plot theoretical distribution
    x_range = np.linspace(0, max(samples_np) * 1.1, 1000)

    if dist_name == "rayleigh":
        theoretical_pdf = stats.rayleigh.pdf(x_range, **theoretical_params)
    elif dist_name == "nakagami":
        theoretical_pdf = stats.nakagami.pdf(x_range, **theoretical_params)

    plt.plot(
        x_range, theoretical_pdf, "r-", linewidth=2, label="Theoretical Distribution"
    )

    # Styling
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(
        f"{dist_name.capitalize()} Distribution Comparison",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Remove top and right spines
    sns.despine()

    # Tight layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# Generate and save the comparison plot
plot_save_path = path.join(results_dir, "distribution_comparison.png")
plot_distribution_comparison(samples, DATASET, theoretical_params, plot_save_path)
print(f"Saved distribution comparison plot to {plot_save_path}")
