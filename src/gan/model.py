import torch
import torch.nn as nn
import lightning as L

torch.set_float32_matmul_precision('medium')

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, z, scale=1):
        return scale*self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class GAN(L.LightningModule):
    def __init__(self, latent_dim=100, g_every_k_steps=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()
        self.automatic_optimization = False  # manual optimization
        self.g_every_k_steps = g_every_k_steps

    def forward(self, z, scale=1):
        return self.generator(z, scale)

    def adversarial_loss(self, y_hat, y):
        return nn.BCELoss()(y_hat, y)

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()

        # =====================
        # Train Discriminator
        # =====================
        z = torch.randn(batch.shape[0], self.latent_dim, device=self.device)
        generated_data = self(z).detach()  # detach so G is not updated here
        data = torch.cat((batch, generated_data))

        real_labels = torch.ones(batch.shape[0], 1, device=self.device)
        generated_labels = torch.zeros(batch.shape[0], 1, device=self.device)
        labels = torch.cat((real_labels, generated_labels))

        d_loss = self.adversarial_loss(self.discriminator(data), labels)

        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()
        self.log("d_loss", d_loss, prog_bar=True, on_step=True, on_epoch=False)

        # =====================
        # Train Generator (every k steps)
        # =====================
        if (batch_idx + 1) % self.g_every_k_steps == 0:
            z = torch.randn(batch.shape[0], self.latent_dim, device=self.device)
            generated_data = self(z)
            valid_labels = torch.ones(batch.shape[0], 1, device=self.device)

            g_loss = self.adversarial_loss(
                self.discriminator(generated_data), valid_labels
            )
            g_opt.zero_grad()
            self.manual_backward(g_loss)
            g_opt.step()
            self.log("g_loss", g_loss, prog_bar=True, on_step=True, on_epoch=False)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        return opt_g, opt_d
    
    def validation_step(self, batch, batch_idx):
        # Generate fake samples
        z = torch.randn(batch.shape[0], self.latent_dim, device=self.device)
        fake_samples = self(z).detach()

        with torch.no_grad():
            real_probs = self.discriminator(batch).mean().item()
            fake_probs = self.discriminator(fake_samples).mean().item()

        self.log("real_prob", real_probs, prog_bar=True)
        self.log("fake_prob", fake_probs, prog_bar=True)

