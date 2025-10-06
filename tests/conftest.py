"""Test configuration and fixtures."""

import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    """Provide device for testing (CPU by default)."""
    return torch.device("cpu")


@pytest.fixture
def random_seed():
    """Set random seeds for reproducible tests."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def latent_dim():
    """Standard latent dimension for testing."""
    return 10  # Small for faster testing


@pytest.fixture
def batch_size():
    """Standard batch size for testing."""
    return 8  # Small for faster testing


@pytest.fixture
def num_samples():
    """Standard number of samples for testing datasets."""
    return 100
