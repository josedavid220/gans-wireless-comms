import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
        self, latent_dim: int = 100, cond_dim: int = 1, cond_emb_dim: int = 16
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.cond_dim = int(cond_dim)
        self.cond_emb_dim = int(cond_emb_dim)

        self.embedding = nn.Sequential(
            nn.Linear(cond_dim, cond_emb_dim),
            nn.ReLU(),
            nn.Linear(cond_emb_dim, cond_emb_dim),
        )

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim + self.cond_emb_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Softplus(beta=1),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(cond)
        x = torch.cat([z, emb], dim=1)

        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, cond_dim: int = 1, cond_emb_dim: int = 16):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.cond_emb_dim = int(cond_emb_dim)

        self.embedding = nn.Sequential(
            nn.Linear(cond_dim, cond_emb_dim),
            nn.ReLU(),
            nn.Linear(cond_emb_dim, cond_emb_dim),
        )
        self.model = nn.Sequential(
            nn.Linear(1 + self.cond_emb_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(cond)
        x_in = torch.cat([x, emb], dim=1)

        return self.model(x_in)
