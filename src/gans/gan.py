import torch
import torch.nn as nn
from .components import Generator, Discriminator
from .base_gan import BaseGAN
import torch.nn.functional as F

torch.set_float32_matmul_precision("medium")


class GAN(BaseGAN):
    def __init__(
        self,
        latent_dim=100,
        g_every_k_steps=1,
        distribution_name=None,
        distribution_params=None,
        num_test_samples=1000,
    ):
        super().__init__(
            latent_dim=latent_dim,
            g_every_k_steps=g_every_k_steps,
            distribution_name=distribution_name,
            distribution_params=distribution_params,
            num_test_samples=num_test_samples,
        )
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()
        self.criterion = nn.BCEWithLogitsLoss()  # more stable than BCELoss with sigmoid
        # TODO: make these hyperparams configurable
        self.lr = 0.0002
        self.betas = (0.5, 0.999)

        self.save_hyperparameters()
        self.apply(self._init_weights)

    def forward(self, z):
        return self.generator(z)

    def compute_discriminator_loss(self, real_samples, fake_samples):
        real_logits = self.discriminator(real_samples)
        fake_logits = self.discriminator(fake_samples)
        logits = torch.cat([real_logits, fake_logits], dim=0)
        labels = torch.cat(
            [torch.ones_like(real_logits), torch.zeros_like(fake_logits)], dim=0
        )

        return self.criterion(logits, labels)

    def compute_generator_loss(self, fake_samples):
        fake_logits = self.discriminator(fake_samples)
        return self.criterion(fake_logits, torch.ones_like(fake_logits))

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=self.betas
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=self.betas
        )
        return opt_g, opt_d

    def validation_step(self, batch, batch_idx):
        z = torch.randn(batch.shape[0], self.latent_dim, device=self.device)
        fake_samples = self(z).detach()

        with torch.no_grad():
            real_prob = F.sigmoid(self.discriminator(batch)).mean().item()
            fake_prob = F.sigmoid(self.discriminator(fake_samples)).mean().item()

        self.log("real_prob", real_prob, prog_bar=True)
        self.log("fake_prob", fake_prob, prog_bar=True)
