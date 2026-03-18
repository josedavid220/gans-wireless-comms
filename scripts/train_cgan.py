import json
import os

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
import torch
import numpy as np

from cgans import CGAN
from local_datasets import MftrUniformConditionalDataset
from config import get_cgan_args
from cgans.mftr_evaluation import MftrRanges, run_end_of_training_evaluation


def _parse_range(value: str, name: str) -> tuple[float, float]:
    parsed = json.loads(value)
    if (
        not isinstance(parsed, list)
        or len(parsed) != 2
        or not all(isinstance(x, (int, float)) for x in parsed)
    ):
        raise ValueError(f"{name} must be a JSON array like [low, high]. Got: {value}")
    return float(parsed[0]), float(parsed[1])


def _sample_conds(
    *,
    combos: int,
    m_range: tuple[float, float],
    mu_range: tuple[float, float],
    K_range: tuple[float, float],
    delta_range: tuple[float, float],
    omega_range: tuple[float, float],
    seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    conds = np.empty((int(combos), 5), dtype=np.float32)

    def u(low: float, high: float) -> float:
        return low if low == high else float(rng.uniform(low, high))

    for i in range(int(combos)):
        conds[i] = np.array(
            [
                u(*m_range),
                u(*mu_range),
                u(*K_range),
                u(*delta_range),
                u(*omega_range),
            ],
            dtype=np.float32,
        )

    return torch.tensor(conds, dtype=torch.float32)


def main():
    args = get_cgan_args()

    m_range = _parse_range(args.m_range, "--m_range")
    mu_range = _parse_range(args.mu_range, "--mu_range")
    K_range = _parse_range(args.K_range, "--K_range")
    delta_range = _parse_range(args.delta_range, "--delta_range")
    omega_range = _parse_range(args.omega_range, "--omega_range")
    use_disjoint_split = args.val_combos is not None
    val_combos = 0 if args.val_combos is None else int(args.val_combos)
    if val_combos < 0:
        raise ValueError("--val_combos must be >= 0")

    L.seed_everything(seed=args.seed, workers=True)

    model = CGAN(
        latent_dim=args.latent_dim,
        cond_dim=args.cond_dim,
        g_every_k_steps=args.g_every_k_steps,
        distribution_name="mftr",
        val_metric_max_samples=int(args.val_metric_max_samples),
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        betas_g=tuple(args.betas_g),
        betas_d=tuple(args.betas_d),
    )

    if use_disjoint_split and val_combos > 0:
        total_combos = int(args.combos) + int(val_combos)
        all_conds = _sample_conds(
            combos=total_combos,
            m_range=m_range,
            mu_range=mu_range,
            K_range=K_range,
            delta_range=delta_range,
            omega_range=omega_range,
            seed=int(args.seed),
        )

        perm = np.random.default_rng(int(args.seed)).permutation(total_combos)
        train_idx = perm[: int(args.combos)]
        val_idx = perm[int(args.combos) :]

        train_conds = all_conds[torch.tensor(train_idx, dtype=torch.long)]
        val_conds = all_conds[torch.tensor(val_idx, dtype=torch.long)]

        dataset = MftrUniformConditionalDataset(
            samples_per_combo=args.samples_per_combo,
            combos=train_conds.shape[0],
            conds_raw=train_conds,
            m=m_range,
            mu=mu_range,
            K=K_range,
            delta=delta_range,
            omega=omega_range,
            normalize_conds=bool(args.normalize_conds),
            dist_type=args.dist_type,
            seed=args.seed,
        )

        val_dataset = MftrUniformConditionalDataset(
            samples_per_combo=args.val_samples_per_combo,
            combos=val_conds.shape[0],
            conds_raw=val_conds,
            m=m_range,
            mu=mu_range,
            K=K_range,
            delta=delta_range,
            omega=omega_range,
            normalize_conds=bool(args.normalize_conds),
            dist_type=args.dist_type,
            seed=args.seed + 1,
        )
    else:
        dataset = MftrUniformConditionalDataset(
            samples_per_combo=args.samples_per_combo,
            combos=args.combos,
            m=m_range,
            mu=mu_range,
            K=K_range,
            delta=delta_range,
            omega=omega_range,
            normalize_conds=bool(args.normalize_conds),
            dist_type=args.dist_type,
            seed=args.seed,
        )

        val_dataset = MftrUniformConditionalDataset(
            samples_per_combo=args.val_samples_per_combo,
            combos=int(args.combos) if args.val_combos is None else int(val_combos),
            m=m_range,
            mu=mu_range,
            K=K_range,
            delta=delta_range,
            omega=omega_range,
            normalize_conds=bool(args.normalize_conds),
            dist_type=args.dist_type,
            seed=args.seed + 1,
        )

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=len(val_dataset),
        num_workers=args.num_workers,
        shuffle=False,
    )


    logger = TensorBoardLogger(save_dir=args.logs_dir, name="mftr/cgan")

    callbacks = []
    # Early stopping (commented on purpose):
    # We first want to observe how `val_wasserstein` evolves across epochs.
    
    # callbacks.append(
    #     EarlyStopping(
    #         monitor="val_wasserstein",
    #         mode="min",
    #         patience=20,
    #         min_delta=0.0,
    #         check_on_train_epoch_end=False,
    #     )
    # )
    
    # callbacks.append(
    #     ModelCheckpoint(
    #         monitor="val_wasserstein",
    #         mode="min",
    #         filename="best_model",
    #         save_top_k=1,
    #         save_last=True,
    #     )
    # )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        strategy="ddp_find_unused_parameters_true",
        accelerator="auto",
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        benchmark=False,
        log_every_n_steps=50,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # -----------------------
    # End-of-training evaluation (rank-0 only)
    # -----------------------
    if trainer.is_global_zero and trainer.logger and trainer.logger.log_dir:
        out_dir = os.path.join(trainer.logger.log_dir, "test_results")
        os.makedirs(out_dir, exist_ok=True)

        # Save conditioning vectors for reproducibility
        torch.save(
            val_dataset.conds_raw.detach().cpu(), os.path.join(out_dir, "val_conds_raw.pt")
        )

        ranges = MftrRanges(
            m=m_range,
            mu=mu_range,
            K=K_range,
            delta=delta_range,
            omega=omega_range,
        )

        run_end_of_training_evaluation(
            out_dir=out_dir,
            model=model,
            train_dataset=dataset,
            ranges=ranges,
            dist_type=args.dist_type,
            seed=int(args.seed),
            eval_max_mixture_samples=int(args.eval_max_mixture_samples),
            eval_num_params_in=int(args.eval_num_params_in),
            eval_num_params_out=int(args.eval_num_params_out),
            eval_num_samples_per_param=int(args.eval_num_samples_per_param),
            normalize_conds=bool(args.normalize_conds),
            eval_save_per_experiment_images=bool(args.eval_save_per_experiment_images),
            batch_size=int(args.batch_size),
        )


if __name__ == "__main__":
    main()
