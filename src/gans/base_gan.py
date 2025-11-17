import torch
import lightning as L
from abc import ABC, abstractmethod
import torch.nn.init as init
import torch.nn as nn
import json
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class BaseGAN(L.LightningModule, ABC):
    def __init__(
        self,
        latent_dim=100,
        g_every_k_steps=1,
        distribution_name=None,
        distribution_params=None,
        num_test_samples=1000,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.g_every_k_steps = g_every_k_steps
        self.automatic_optimization = False
        self.distribution_name = distribution_name
        self.distribution_params = distribution_params or {}
        self.num_test_samples = num_test_samples

        self.generator: nn.Module
        self.discriminator: nn.Module

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)

    @abstractmethod
    def compute_discriminator_loss(self, real_samples, fake_samples) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_generator_loss(self, fake_samples) -> torch.Tensor:
        pass

    def forward(self, z, *args):
        return self.generator(z, *args)

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()  # type: ignore

        z = torch.randn(batch.shape[0], self.latent_dim, device=self.device)
        generated_data = self(z).detach()

        d_loss = self.compute_discriminator_loss(batch, generated_data)
        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()
        self.log("d_loss", d_loss, prog_bar=True, on_step=True, on_epoch=False)

        if (batch_idx + 1) % self.g_every_k_steps == 0:
            z = torch.randn(batch.shape[0], self.latent_dim, device=self.device)
            generated_data = self(z)
            g_loss = self.compute_generator_loss(generated_data)

            # freeze D while updating G
            for p in self.discriminator.parameters():
                p.requires_grad_(False)

            g_opt.zero_grad()
            self.manual_backward(g_loss)
            g_opt.step()

            for p in self.discriminator.parameters():
                p.requires_grad_(True)

            self.log("g_loss", g_loss, prog_bar=True, on_step=True, on_epoch=False)

    def test_step(self, batch, batch_idx):
        """Perform goodness-of-fit tests and generate comparison plots."""
        if not self.distribution_name:
            return

        # Generate samples for testing
        with torch.no_grad():
            z = torch.randn(self.num_test_samples, self.latent_dim, device=self.device)
            samples = self(z).squeeze().cpu().numpy()

        # Perform GOF tests
        results = self._perform_gof_tests(samples)

        # Log metrics
        for test_name, test_result in results["tests"].items():
            self.log(f"gof_{test_name}_pvalue", test_result["pvalue"])

        self._save_test_artifacts(samples, results)

    def _perform_gof_tests(self, samples):
        """Perform goodness-of-fit tests."""
        tests = ["ks", "ad"]
        results = {
            "theoretical_distribution": self.distribution_name,
            "theoretical_params": self.distribution_params,
            "gan_type": type(self).__name__,
            "generated_samples": len(samples),
            "tests": {},
        }

        for test in tests:
            result = self._compute_gof_test(samples, self.distribution_name, test)
            results["tests"][test] = {
                "statistic": float(result.statistic),
                "pvalue": float(result.pvalue),
                "estimated_params": result.fit_result.params._asdict(),
            }

        return results

    def _compute_gof_test(self, samples, dist_name, statistic="ks", n_mc_samples=1000):
        """Compute a single goodness-of-fit test."""
        if dist_name == "rayleigh":
            dist = stats.rayleigh
        elif dist_name == "nakagami":
            dist = stats.nakagami
        else:
            raise ValueError(f"Unsupported distribution: {dist_name}")

        rng = np.random.default_rng(0)
        return stats.goodness_of_fit(
            dist=dist,
            data=samples,
            statistic=statistic,
            n_mc_samples=n_mc_samples,
            rng=rng,
        )

    def _save_test_artifacts(self, samples, results):
        """Save test results and comparison plot."""
        # Create results directory at the same level as checkpoints
        if (
            not self.trainer
            or not self.trainer.logger
            or not self.trainer.logger.log_dir
        ):
            print(
                "Warning: Cannot save test artifacts - logger or log_dir not available"
            )
            return

        log_dir = self.trainer.logger.log_dir
        test_results_dir = os.path.join(log_dir, "test_results")
        os.makedirs(test_results_dir, exist_ok=True)

        # Save JSON results
        json_path = os.path.join(test_results_dir, "gof_tests.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)

        # Generate and save comparison plot
        plot_path = os.path.join(test_results_dir, "distribution_comparison.png")
        self._plot_distribution_comparison(samples, plot_path)

        print(f"Saved test results to {json_path}")
        print(f"Saved distribution comparison plot to {plot_path}")

    def _plot_distribution_comparison(self, samples, save_path):
        """Generate comparison plot showing histogram, KDE, and theoretical distribution."""
        if not self.distribution_name:
            return

        plt.figure(figsize=(10, 6), dpi=300)

        # Histogram and KDE of generated samples
        sns.histplot(
            samples,
            alpha=0.6,
            color="skyblue",
            label=f"Generated Samples ({type(self).__name__})",
            stat="density",
        )
        sns.kdeplot(samples, color="darkblue", linewidth=2, label="KDE")

        # Plot theoretical distribution
        x_range = np.linspace(0, max(samples) * 1.1, 1000)
        theoretical_pdf = None

        if self.distribution_name == "rayleigh":
            theoretical_pdf = stats.rayleigh.pdf(x_range, **self.distribution_params)
        elif self.distribution_name == "nakagami":
            theoretical_pdf = stats.nakagami.pdf(x_range, **self.distribution_params)

        if theoretical_pdf is not None:
            # Format distribution parameters for display
            params_str = ", ".join(
                [f"{k}={v:.2f}" for k, v in self.distribution_params.items()]
            )

            plt.plot(
                x_range,
                theoretical_pdf,
                "r-",
                linewidth=2,
                label=f"Theoretical {self.distribution_name} ({params_str})",
            )

        # Styling
        plt.xlabel("Value", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.title(
            f"{self.distribution_name.capitalize()} Distribution Comparison",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        sns.despine()
        plt.tight_layout()

        # Log the figure to the logger before saving
        self.trainer.logger.experiment.add_figure(  # type: ignore
            "distribution_comparison",
            plt.gcf(),
            close=False,
        )

        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
