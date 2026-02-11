import json

import lightning as L
import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from cgans import CGAN
from cgans.mftr_plotting import save_mftr_conditional_comparison_figure
from local_datasets import MftrConditionalDataset
from config import get_cgan_args


def main():
    args = get_cgan_args()

    m_values = json.loads(args.m_values)
    mu_values = json.loads(args.mu_values)
    K_values = json.loads(args.K_values)
    delta_values = json.loads(args.delta_values)
    omega_values = json.loads(args.omega_values)

    L.seed_everything(seed=args.seed, workers=True)

    model = CGAN(
        latent_dim=args.latent_dim,
        cond_dim=args.cond_dim,
        g_every_k_steps=args.g_every_k_steps,
        distribution_name="mftr",
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        betas_g=tuple(args.betas_g),
        betas_d=tuple(args.betas_d),
    )

    dataset = MftrConditionalDataset(
        samples_per_combo=args.samples_per_combo,
        param_grid={
            "m": m_values,
            "mu": mu_values,
            "K": K_values,
            "delta": delta_values,
            "omega": omega_values,
        },
        seed=args.seed,
    )

    val_dataset = MftrConditionalDataset(
        samples_per_combo=args.val_samples_per_combo,
        param_grid={
            "m": m_values,
            "mu": mu_values,
            "K": K_values,
            "delta": delta_values,
            "omega": omega_values,
        },
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

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        logger=logger,
        deterministic=True,
        benchmark=False,
        log_every_n_steps=int(np.ceil(len(dataset) / args.batch_size)),
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    save_mftr_conditional_comparison_figure(
        trainer=trainer,
        model=model,
        conds_raw=dataset.conds_raw,
        num_samples=min(1000, args.samples_per_combo),
        max_cols=2,
    )


if __name__ == "__main__":
    main()
