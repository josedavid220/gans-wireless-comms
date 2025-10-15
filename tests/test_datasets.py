"""Tests for local dataset classes."""

import torch
from local_datasets import RayleighDataset, NakagamiDataset


class TestRayleighDataset:
    """Test cases for RayleighDataset."""

    def test_initialization(self, num_samples):
        """Test that RayleighDataset initializes correctly."""
        scale = 2.0
        dataset = RayleighDataset(num_samples=num_samples, scale=scale)

        assert dataset.num_samples == num_samples
        assert dataset.scale == scale
        assert len(dataset) == num_samples
        assert dataset.samples.shape == (num_samples, 1)

    def test_getitem(self, num_samples):
        """Test that __getitem__ returns correct samples."""
        dataset = RayleighDataset(num_samples=num_samples)

        # Test valid indices
        for i in range(min(10, num_samples)):
            sample = dataset[i]
            assert isinstance(sample, torch.Tensor)
            assert sample.shape == (1,)
            assert sample.dtype == torch.float32

    def test_reproducibility(self, num_samples, random_seed):
        """Test that the dataset generates reproducible results."""
        scale = 1.5
        dataset1 = RayleighDataset(
            num_samples=num_samples, scale=scale, seed=random_seed
        )
        dataset2 = RayleighDataset(
            num_samples=num_samples, scale=scale, seed=random_seed
        )

        # Should be identical due to fixed seed
        torch.testing.assert_close(dataset1.samples, dataset2.samples)


class TestNakagamiDataset:
    """Test cases for NakagamiDataset."""

    def test_initialization(self, num_samples):
        """Test that NakagamiDataset initializes correctly."""
        nu = 2.0
        scale = 1.5
        dataset = NakagamiDataset(num_samples=num_samples, nu=nu, scale=scale)

        assert dataset.num_samples == num_samples
        assert dataset.nu == nu
        assert dataset.scale == scale
        assert len(dataset) == num_samples
        assert dataset.samples.shape == (num_samples, 1)

    def test_getitem(self, num_samples):
        """Test that __getitem__ returns correct samples."""
        dataset = NakagamiDataset(num_samples=num_samples)

        # Test valid indices
        for i in range(min(10, num_samples)):
            sample = dataset[i]
            assert isinstance(sample, torch.Tensor)
            assert sample.shape == (1,)
            assert sample.dtype == torch.float32

    def test_reproducibility(self, num_samples, random_seed):
        """Test that the dataset generates reproducible results."""
        nu = 1.5
        scale = 2.0
        dataset1 = NakagamiDataset(
            num_samples=num_samples, nu=nu, scale=scale, seed=random_seed
        )
        dataset2 = NakagamiDataset(
            num_samples=num_samples, nu=nu, scale=scale, seed=random_seed
        )

        # Should be identical due to fixed seed
        torch.testing.assert_close(dataset1.samples, dataset2.samples)
