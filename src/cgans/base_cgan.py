import torch
import lightning as L
from abc import ABC, abstractmethod
import torch.nn.init as init
import torch.nn as nn


class BaseCGAN(L.LightningModule, ABC):
    def __init__(
        self,
        latent_dim=100,
        g_every_k_steps=1,
        distribution_name=None,
        num_test_samples=1000,
        lr_g=0.0002,
        lr_d=0.0002,
        betas_g=(0.5, 0.999),
        betas_d=(0.5, 0.999),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.g_every_k_steps = g_every_k_steps
        self.automatic_optimization = False
        self.distribution_name = distribution_name
        self.num_test_samples = num_test_samples
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.betas_g = betas_g
        self.betas_d = betas_d

        self.generator: nn.Module
        self.discriminator: nn.Module

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)

    @abstractmethod
    def compute_discriminator_loss(
        self,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_generator_loss(
        self, fake_samples: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        pass

    def forward(self, z: torch.Tensor, cond: torch.Tensor, *args):
        return self.generator(z, cond, *args)

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()  # type: ignore

        real_samples, cond = batch

        z = torch.randn(real_samples.shape[0], self.latent_dim, device=self.device)
        generated_data = self(z, cond).detach()

        d_loss = self.compute_discriminator_loss(real_samples, generated_data, cond)
        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()
        self.log("d_loss", d_loss, prog_bar=True, on_step=True, on_epoch=False)

        if (batch_idx + 1) % self.g_every_k_steps == 0:
            z = torch.randn(real_samples.shape[0], self.latent_dim, device=self.device)
            generated_data = self(z, cond)
            g_loss = self.compute_generator_loss(generated_data, cond)

            # freeze D while updating G
            for p in self.discriminator.parameters():
                p.requires_grad_(False)

            g_opt.zero_grad()
            self.manual_backward(g_loss)
            g_opt.step()

            for p in self.discriminator.parameters():
                p.requires_grad_(True)

            self.log("g_loss", g_loss, prog_bar=True, on_step=True, on_epoch=False)
