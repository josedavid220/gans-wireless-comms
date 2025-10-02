import torch
import lightning as L
from abc import ABC, abstractmethod


class BaseGAN(L.LightningModule, ABC):
    def __init__(self, latent_dim=100, g_every_k_steps=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.g_every_k_steps = g_every_k_steps
        self.automatic_optimization = False

    @abstractmethod
    def compute_discriminator_loss(self, real_samples, fake_samples):
        pass

    @abstractmethod
    def compute_generator_loss(self, fake_samples):
        pass

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()

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
            g_opt.zero_grad()
            self.manual_backward(g_loss)
            g_opt.step()
            self.log("g_loss", g_loss, prog_bar=True, on_step=True, on_epoch=False)
