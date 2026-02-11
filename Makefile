.PHONY: test lint format build

train-rayleigh:
	cd scripts && \
	uv run python train.py \
	--epochs 150 \
	--distribution_params '{"scale": 1.0}' \
	--distribution rayleigh \
	--gan_type wgan_gp \
	--num_workers 15 \
	--num_test_samples 2000 \
	--batch_size \
	-k 5

train-nakagami:
	cd scripts && \
	uv run python train.py \
	--epochs 100 \
	--distribution_params '{"nu": 3.1415, "scale": 10}' \
	--distribution nakagami \
	--num_workers 15 \
	--gan_type gan \
	--num_test_samples 5000 \
	--batch_size 256

train-mftr:
	cd scripts && \
	uv run python train.py \
	--epochs 160 \
	--distribution_params '{"m": 8, "mu": 3, "K": 8.0, "delta": 0.9, "omega": 2.0}' \
	--distribution mftr \
	--num_workers 15 \
	--gan_type gan \
	--num_test_samples 2000 \
	--batch_size 256 \
	-d 1028 \
	-k 5 \
	--betas_g 0.6 0.999 \
	--num_samples 20000

train-cgan-mftr:
	cd scripts && \
	uv run python train_cgan.py \
	--epochs 150 \
	--batch_size 256 \
	--latent_dim 1024 \
	--g_every_k_steps 5 \
	--cond_dim 5 \
	--cond_emb_dim 32 \
	--samples_per_combo 1000 \
	--val_samples_per_combo 1000 \
	--m_values "[5, 6, 7, 8]" \
	--mu_values "[1, 2, 3, 4, 5, 6, 7, 8]" \
	--K_values "[8, 9, 10]" \
	--delta_values "[0.1, 0.5, 0.9]" \
	--omega_values "[1, 1.5, 2]" \
	--num_workers 10 \
	--betas_g 0.6 0.999

test-gan:
	cd scripts && \
	uv run python test.py \
	--distribution nakagami \
	--num_test_samples 2000 \
	--num_workers 15 \
	--gan_type gan \
	-v 4

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts

format:
	uv run ruff format src tests scripts

build:
	uv build

install-hooks:
	uv run pre-commit install

run-hooks:
	uv run pre-commit run --all-files

tensorboard:
	tensorboard --logdir logs/mftr/cgan

clean:
	rm -rf logs/* || true && rm -rf results/tests/* || true
