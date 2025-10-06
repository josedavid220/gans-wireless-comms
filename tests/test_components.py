"""Tests for GAN components (Generator, Discriminator, Critic)."""

import torch

from gans.components import Generator, Discriminator, Critic


class TestGenerator:
    """Test cases for Generator class."""

    def test_forward_pass(self, latent_dim, batch_size, device):
        """Test forward pass through generator."""
        generator = Generator(latent_dim=latent_dim).to(device)
        z = torch.randn(batch_size, latent_dim, device=device)

        with torch.no_grad():
            output = generator(z)

        assert output.shape == (batch_size, 1)
        assert output.device == device
        assert output.dtype == torch.float32

class TestDiscriminator:
    """Test cases for Discriminator class."""

    def test_forward_pass(self, batch_size, device):
        """Test forward pass through discriminator."""
        discriminator = Discriminator().to(device)
        x = torch.randn(batch_size, 1, device=device)

        with torch.no_grad():
            output = discriminator(x)

        assert output.shape == (batch_size, 1)
        assert output.device == device
        assert output.dtype == torch.float32


class TestCritic:
    """Test cases for Critic class."""

    def test_forward_pass(self, batch_size, device):
        """Test forward pass through critic."""
        critic = Critic().to(device)
        x = torch.randn(batch_size, 1, device=device)

        with torch.no_grad():
            output = critic(x)

        assert output.shape == (batch_size, 1)
        assert output.device == device
        assert output.dtype == torch.float32


class TestComponentsIntegration:
    """Integration tests for components working together."""

    def test_generator_discriminator_compatibility(self, latent_dim, batch_size):
        """Test that generator output is compatible with discriminator input."""
        generator = Generator(latent_dim=latent_dim)
        discriminator = Discriminator()

        z = torch.randn(batch_size, latent_dim)

        with torch.no_grad():
            fake_data = generator(z)
            disc_output = discriminator(fake_data)

        assert fake_data.shape == (batch_size, 1)
        assert disc_output.shape == (batch_size, 1)

    def test_generator_critic_compatibility(self, latent_dim, batch_size):
        """Test that generator output is compatible with critic input."""
        generator = Generator(latent_dim=latent_dim)
        critic = Critic()

        z = torch.randn(batch_size, latent_dim)

        with torch.no_grad():
            fake_data = generator(z)
            critic_output = critic(fake_data)

        assert fake_data.shape == (batch_size, 1)
        assert critic_output.shape == (batch_size, 1)
