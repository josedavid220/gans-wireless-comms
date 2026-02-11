"""Tests for local dataset classes."""

import torch
from local_datasets import RayleighDataset, NakagamiDataset, MftrConditionalDataset


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


class TestMftrConditionalDataset:
    def test_initialization(self):
        samples_per_combo = 8
        param_grid = {"mu": [2, 8], "delta": [0.2, 0.8]}
        dataset = MftrConditionalDataset(
            samples_per_combo=samples_per_combo,
            param_grid=param_grid,
            m=8,
            K=8.0,
            omega=2.0,
            seed=0,
        )

        assert dataset.samples_per_combo == samples_per_combo
        assert dataset.conds_raw.shape == (4, 5)
        assert len(dataset) == 4 * samples_per_combo
        assert dataset.samples.shape == (4 * samples_per_combo, 1)

    def test_getitem(self):
        samples_per_combo = 5
        param_grid = {"mu": [2, 8], "delta": [0.2, 0.8]}
        dataset = MftrConditionalDataset(
            samples_per_combo=samples_per_combo,
            param_grid=param_grid,
            m=8,
            K=8.0,
            omega=2.0,
            seed=0,
        )

        x0, c0 = dataset[0]
        assert isinstance(x0, torch.Tensor)
        assert x0.shape == (1,)
        assert x0.dtype == torch.float32

        assert isinstance(c0, torch.Tensor)
        assert c0.shape == (5,)
        assert c0.dtype == torch.float32

        # First combo follows product order over keys ['mu', 'delta']
        # cond order is [m, mu, K, delta, omega]
        expected = torch.tensor([8.0, 2.0, 8.0, 0.2, 2.0], dtype=torch.float32)
        torch.testing.assert_close(c0, expected)

        # Item at the start of second combo should have same cond, different sample index
        x1, c1 = dataset[samples_per_combo]
        torch.testing.assert_close(c1, dataset.conds_raw[1])
        assert x1.shape == (1,)

    def test_reproducibility(self):
        samples_per_combo = 6
        param_grid = {"mu": [2, 8], "delta": [0.2, 0.8]}

        dataset1 = MftrConditionalDataset(
            samples_per_combo=samples_per_combo,
            param_grid=param_grid,
            m=8,
            K=8.0,
            omega=2.0,
            seed=123,
        )
        dataset2 = MftrConditionalDataset(
            samples_per_combo=samples_per_combo,
            param_grid=param_grid,
            m=8,
            K=8.0,
            omega=2.0,
            seed=123,
        )

        torch.testing.assert_close(dataset1.samples, dataset2.samples)
        torch.testing.assert_close(dataset1.conds_raw, dataset2.conds_raw)
