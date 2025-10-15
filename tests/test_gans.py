"""Tests for GAN classes."""

import torch
from unittest.mock import patch
import lightning as L

from src.gans.gan import GAN
from src.gans.wgan_gp import WGAN_GP
from lightning.pytorch.utilities import disable_possible_user_warnings


class TestGAN:
    """Test cases for GAN class."""

    def test_initialization(self, latent_dim):
        """Test that GAN initializes correctly."""
        gan = GAN(latent_dim=latent_dim, g_every_k_steps=2)
        assert hasattr(gan, "generator")
        assert hasattr(gan, "discriminator")

    def test_forward_pass(self, latent_dim, batch_size, device):
        """Test forward pass through GAN."""
        gan = GAN(latent_dim=latent_dim).to(device)
        z = torch.randn(batch_size, latent_dim, device=device)

        with torch.no_grad():
            output = gan(z)

        assert output.shape == (batch_size, 1)
        assert output.device == device

    def test_compute_discriminator_loss(self, latent_dim, batch_size, device):
        """Test discriminator loss computation."""
        gan = GAN(latent_dim=latent_dim).to(device)

        real_samples = torch.randn(batch_size, 1, device=device)
        fake_samples = torch.randn(batch_size, 1, device=device)

        with torch.no_grad():
            loss = gan.compute_discriminator_loss(real_samples, fake_samples)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar
        assert loss >= 0  # BCE loss should be non-negative

    def test_compute_generator_loss(self, latent_dim, batch_size, device):
        """Test generator loss computation."""
        gan = GAN(latent_dim=latent_dim).to(device)

        fake_samples = torch.randn(batch_size, 1, device=device)

        with torch.no_grad():
            loss = gan.compute_generator_loss(fake_samples)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar
        assert loss >= 0  # BCE loss should be non-negative

    def test_one_epoch_training(self, latent_dim, train_dataloader):
        """Test that GAN can complete one epoch of training."""
        gan = GAN(latent_dim=latent_dim, g_every_k_steps=2)

        # Create a minimal trainer
        disable_possible_user_warnings()
        trainer = L.Trainer(
            max_epochs=1,
            accelerator="auto",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        # This should complete without errors
        trainer.fit(model=gan, train_dataloaders=train_dataloader)


class TestWGAN_GP:
    """Test cases for WGAN_GP class."""

    def test_initialization(self, latent_dim):
        """Test that WGAN_GP initializes correctly."""
        lambda_gp = 15.0
        g_every_k_steps = 3
        wgan = WGAN_GP(
            latent_dim=latent_dim, g_every_k_steps=g_every_k_steps, lambda_gp=lambda_gp
        )

        assert hasattr(wgan, "generator")
        assert hasattr(wgan, "discriminator")

    def test_forward_pass(self, latent_dim, batch_size, device):
        """Test forward pass through WGAN_GP."""
        wgan = WGAN_GP(latent_dim=latent_dim).to(device)
        z = torch.randn(batch_size, latent_dim, device=device)

        with torch.no_grad():
            output = wgan(z)

        assert output.shape == (batch_size, 1)
        assert output.device == device

    def test_compute_gradient_penalty(self, latent_dim, batch_size, device):
        """Test gradient penalty computation."""
        wgan = WGAN_GP(latent_dim=latent_dim, lambda_gp=10.0).to(device)

        real_samples = torch.randn(batch_size, 1, device=device)
        fake_samples = torch.randn(batch_size, 1, device=device)

        gp = wgan.compute_gradient_penalty(real_samples, fake_samples)

        assert isinstance(gp, torch.Tensor)
        assert gp.shape == ()  # Scalar
        assert gp >= 0  # Gradient penalty should be non-negative

    def test_compute_discriminator_loss(self, latent_dim, batch_size, device):
        """Test discriminator loss computation (Wasserstein + GP)."""
        wgan = WGAN_GP(latent_dim=latent_dim).to(device)

        real_samples = torch.randn(batch_size, 1, device=device)
        fake_samples = torch.randn(batch_size, 1, device=device)

        loss = wgan.compute_discriminator_loss(real_samples, fake_samples)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar

    def test_compute_generator_loss(self, latent_dim, batch_size, device):
        """Test generator loss computation (negative Wasserstein)."""
        wgan = WGAN_GP(latent_dim=latent_dim).to(device)

        fake_samples = torch.randn(batch_size, 1, device=device)

        loss = wgan.compute_generator_loss(fake_samples)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar

    def test_lambda_gp_effect(self, latent_dim, batch_size, device):
        """Test that lambda_gp affects the gradient penalty magnitude."""
        real_samples = torch.randn(batch_size, 1, device=device)
        fake_samples = torch.randn(batch_size, 1, device=device)

        wgan1 = WGAN_GP(latent_dim=latent_dim, lambda_gp=1.0).to(device)
        wgan2 = WGAN_GP(latent_dim=latent_dim, lambda_gp=10.0).to(device)

        # Copy weights to ensure same network behavior
        wgan2.load_state_dict(wgan1.state_dict())

        gp1 = wgan1.compute_gradient_penalty(real_samples, fake_samples)
        gp2 = wgan2.compute_gradient_penalty(real_samples, fake_samples)

        # GP2 should be approximately 10x larger than GP1
        ratio = gp2 / gp1
        assert abs(ratio - 10.0) < 1.0  # Allow some tolerance

    def test_gradient_penalty_full(self, latent_dim, batch_size, device):
        """Test the core logic and properties of the gradient penalty."""
        wgan = WGAN_GP(latent_dim=latent_dim, lambda_gp=10.0).to(device)

        # 1. Test with identical samples (interpolated samples are also identical)
        identical_samples = torch.randn(batch_size, 1, device=device)
        # Patch the critic to be a simple sum function so its gradient w.r.t input is 1
        # This makes the gradient norm equal to 1 and the GP approximately 0
        with patch.object(wgan.discriminator, "forward") as mock_disc:
            mock_disc.side_effect = lambda x: torch.sum(x, dim=1, keepdim=True)
            gp_identical = wgan.compute_gradient_penalty(
                identical_samples, identical_samples
            )
            assert torch.isclose(
                gp_identical, torch.tensor(0.0, device=device), atol=1e-4
            )

        # 2. Test that interpolation is happening and the gradient is non-trivial
        real_samples = torch.randn(batch_size, 1, device=device)
        fake_samples = torch.randn(batch_size, 1, device=device)

        # We need to manually check the interpolation. A good way is to
        # mock the interpolation function if it exists, or check the
        # intermediate gradients if possible.
        # A simpler approach is to create inputs with known gradient norms.

        # Mock critic to have a simple, linear gradient.
        with patch.object(wgan.discriminator, "forward") as mock_disc:

            def side_effect(x):
                # The output score should depend on the input to create a gradient.
                return torch.sum(x, dim=1, keepdim=True) * 2.0

            mock_disc.side_effect = side_effect

            # Now, `compute_gradient_penalty` should compute the gradient
            # of the mocked function (which is 2) w.r.t the interpolated samples.
            # The penalty should be based on (|2| - 1)^2 = 1.
            gp_mocked = wgan.compute_gradient_penalty(real_samples, fake_samples)

            assert torch.isclose(
                gp_mocked, torch.tensor(1.0, device=device) * 10.0, atol=1e-4
            )
            # Note: The result is 10.0 because lambda_gp is 10.0

    def test_one_epoch_training(self, latent_dim, train_dataloader):
        """Test that WGAN_GP can complete one epoch of training."""
        wgan = WGAN_GP(latent_dim=latent_dim, g_every_k_steps=2, lambda_gp=10.0)

        # Create a minimal trainer
        disable_possible_user_warnings()
        trainer = L.Trainer(
            max_epochs=1,
            accelerator="auto",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        # This should complete without errors
        trainer.fit(model=wgan, train_dataloaders=train_dataloader)
