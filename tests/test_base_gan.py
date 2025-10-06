"""Tests for BaseGAN abstract base class."""

import pytest
import torch
import lightning as L
from unittest.mock import patch, MagicMock
from abc import ABC

from gans.base_gan import BaseGAN


class ConcreteGAN(BaseGAN):
    """Concrete implementation of BaseGAN for testing."""
    
    def __init__(self, latent_dim=100, g_every_k_steps=1):
        super().__init__(latent_dim=latent_dim, g_every_k_steps=g_every_k_steps)
        # Simple linear layers for testing
        self.generator = torch.nn.Linear(latent_dim, 1)
        self.discriminator = torch.nn.Linear(1, 1)
    
    def forward(self, z):
        return self.generator(z)
    
    def compute_discriminator_loss(self, real_samples, fake_samples):
        # Simple mock loss
        return torch.tensor(1.0, requires_grad=True)
    
    def compute_generator_loss(self, fake_samples):
        # Simple mock loss
        return torch.tensor(0.5, requires_grad=True)
    
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)
        return opt_g, opt_d


class TestBaseGAN:
    """Test cases for BaseGAN abstract base class."""

    def test_concrete_implementation_initialization(self, latent_dim):
        """Test that concrete implementation initializes correctly."""
        g_every_k_steps = 3
        gan = ConcreteGAN(latent_dim=latent_dim, g_every_k_steps=g_every_k_steps)
        
        assert gan.latent_dim == latent_dim
        assert gan.g_every_k_steps == g_every_k_steps
        assert gan.automatic_optimization == False

    def test_inheritance_structure(self, latent_dim):
        """Test that BaseGAN properly inherits from LightningModule and ABC."""
        gan = ConcreteGAN(latent_dim=latent_dim)
        
        assert isinstance(gan, L.LightningModule)
        assert isinstance(gan, ABC)
        assert isinstance(gan, BaseGAN)

    def test_abstract_methods_exist(self):
        """Test that abstract methods are defined."""
        # These should be abstract and implemented in subclasses
        assert hasattr(BaseGAN, 'compute_discriminator_loss')
        assert hasattr(BaseGAN, 'compute_generator_loss')

    @patch('lightning.LightningModule.log')
    def test_training_step_discriminator_only(self, mock_log, latent_dim, batch_size, device):
        """Test training step when generator is not updated (batch_idx not divisible by g_every_k_steps)."""
        gan = ConcreteGAN(latent_dim=latent_dim, g_every_k_steps=2).to(device)
        batch = torch.randn(batch_size, 1, device=device)
        
        # Mock optimizers
        mock_g_opt = MagicMock()
        mock_d_opt = MagicMock()
        
        with patch.object(gan, 'optimizers', return_value=(mock_g_opt, mock_d_opt)):
            with patch.object(gan, 'manual_backward') as mock_backward:
                # batch_idx = 0, not divisible by g_every_k_steps=2
                gan.training_step(batch, batch_idx=0)
        
        # Discriminator should be updated
        mock_d_opt.zero_grad.assert_called_once()
        mock_d_opt.step.assert_called_once()
        
        # Generator should NOT be updated
        mock_g_opt.zero_grad.assert_not_called()
        mock_g_opt.step.assert_not_called()
        
        # Only discriminator loss should be logged
        mock_log.assert_called_once()
        assert mock_log.call_args[0][0] == "d_loss"

    @patch('lightning.LightningModule.log')
    def test_training_step_both_networks(self, mock_log, latent_dim, batch_size, device):
        """Test training step when both networks are updated."""
        gan = ConcreteGAN(latent_dim=latent_dim, g_every_k_steps=2).to(device)
        batch = torch.randn(batch_size, 1, device=device)
        
        # Mock optimizers
        mock_g_opt = MagicMock()
        mock_d_opt = MagicMock()
        
        with patch.object(gan, 'optimizers', return_value=(mock_g_opt, mock_d_opt)):
            with patch.object(gan, 'manual_backward') as mock_backward:
                # batch_idx = 1, (1+1) % 2 == 0, so generator should be updated
                gan.training_step(batch, batch_idx=1)
        
        # Both optimizers should be updated
        mock_d_opt.zero_grad.assert_called_once()
        mock_d_opt.step.assert_called_once()
        mock_g_opt.zero_grad.assert_called_once()
        mock_g_opt.step.assert_called_once()
        
        # Both losses should be logged
        assert mock_log.call_count == 2
        logged_metrics = [call[0][0] for call in mock_log.call_args_list]
        assert "d_loss" in logged_metrics
        assert "g_loss" in logged_metrics

    def test_training_step_generator_frequency(self, latent_dim, batch_size, device):
        """Test that generator is updated at correct frequency."""
        g_every_k_steps = 3
        gan = ConcreteGAN(latent_dim=latent_dim, g_every_k_steps=g_every_k_steps).to(device)
        batch = torch.randn(batch_size, 1, device=device)
        
        mock_g_opt = MagicMock()
        mock_d_opt = MagicMock()
        
        with patch.object(gan, 'optimizers', return_value=(mock_g_opt, mock_d_opt)):
            with patch.object(gan, 'manual_backward'):
                with patch('lightning.LightningModule.log'):
                    # Test multiple batch indices
                    for batch_idx in range(6):
                        mock_g_opt.reset_mock()
                        mock_d_opt.reset_mock()
                        
                        gan.training_step(batch, batch_idx=batch_idx)
                        
                        # Discriminator should always be updated
                        mock_d_opt.step.assert_called_once()
                        
                        # Generator should be updated only when (batch_idx + 1) % g_every_k_steps == 0
                        if (batch_idx + 1) % g_every_k_steps == 0:
                            mock_g_opt.step.assert_called_once()
                        else:
                            mock_g_opt.step.assert_not_called()

    def test_training_step_detach_for_discriminator(self, latent_dim, batch_size, device):
        """Test that generated samples are detached for discriminator training."""
        gan = ConcreteGAN(latent_dim=latent_dim).to(device)
        batch = torch.randn(batch_size, 1, device=device)
        
        detached_samples = []
        
        def capture_discriminator_loss(real, fake):
            detached_samples.append(fake.requires_grad)
            return torch.tensor(1.0, requires_grad=True)
        
        with patch.object(gan, 'compute_discriminator_loss', side_effect=capture_discriminator_loss):
            with patch.object(gan, 'optimizers', return_value=(MagicMock(), MagicMock())):
                with patch.object(gan, 'manual_backward'):
                    with patch('lightning.LightningModule.log'):
                        gan.training_step(batch, batch_idx=0)
        
        # Fake samples should be detached for discriminator training
        assert len(detached_samples) == 1
        assert detached_samples[0] == False  # Should be detached