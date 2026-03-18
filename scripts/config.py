import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Train a GAN to approximate a Rayleigh distribution"
    )

    # ------------------- General  parameters -------------------
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="../logs",
        help="Directory to save logs and model checkpoints",
    )

    # ------------------- Training parameters -------------------
    parser.add_argument(
        "--gan_type",
        type=str,
        default="gan",
        help="Type of GAN to use",
        choices=["gan", "wgan_gp"],
    )

    parser.add_argument(
        "--distribution",
        type=str,
        default="rayleigh",
        help="Name of the target distribution",
        choices=["rayleigh", "nakagami", "mftr"],
    )

    parser.add_argument(
        "--distribution_params",
        type=str,
        default="{'scale': 1.0}",
        help="Parameters of the target distribution in dictionary format",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to generate for the training dataset",
    )

    parser.add_argument(
        "--g_every_k_steps",
        "-k",
        type=int,
        default=3,
        help="Number of steps to wait for updating the generator",
    )

    parser.add_argument(
        "--latent_dim",
        "-d",
        type=int,
        default=1000,
        help="Latent dimension for the generator",
    )

    parser.add_argument(
        "--lambda_gp",
        type=float,
        default=10.0,
        help="Gradient penalty coefficient for WGAN-GP",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of max epochs",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers to use",
    )

    parser.add_argument(
        "--lr_g",
        type=float,
        default=0.0002,
        help="Learning rate for the generator optimizer",
    )

    parser.add_argument(
        "--lr_d",
        type=float,
        default=0.0002,
        help="Learning rate for the discriminator optimizer",
    )

    parser.add_argument(
        "--betas_g",
        type=float,
        nargs=2,
        default=(0.5, 0.999),
        help="Betas for the generator optimizer",
    )

    parser.add_argument(
        "--betas_d",
        type=float,
        nargs=2,
        default=(0.5, 0.999),
        help="Betas for the discriminator optimizer",
    )

    # ------------------- Testing parameters -------------------
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=10000,
        help="Number of samples to generate for the testing dataset",
    )

    parser.add_argument(
        "--version",
        "-v",
        type=str,
        help="Checkpoint version for testing",
    )

    parser.add_argument(
        "--tests_save_path",
        type=str,
        default="../results/tests",
        help="Path to save testing results",
    )

    return parser.parse_args()


def get_cgan_args():
    parser = argparse.ArgumentParser(
        description="Train a CGAN on MFTR (conditional) samples"
    )

    parser.add_argument(
        "--logs_dir",
        type=str,
        default="../logs",
        help="Directory to save logs and model checkpoints",
    )

    parser.add_argument("--latent_dim", "-d", type=int, default=1000)
    parser.add_argument("--cond_dim", type=int, default=5)
    parser.add_argument("--cond_emb_dim", type=int, default=16)
    parser.add_argument(
        "--g_every_k_steps",
        "-k",
        type=int,
        default=3,
        help="Number of steps to wait for updating the generator",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr_g", type=float, default=0.0001)
    parser.add_argument("--lr_d", type=float, default=0.0002)
    parser.add_argument("--betas_g", type=float, nargs=2, default=(0.5, 0.999))
    parser.add_argument("--betas_d", type=float, nargs=2, default=(0.5, 0.999))

    parser.add_argument(
        "--val_metric_max_samples",
        type=int,
        default=20000,
        help="Max validation samples (pooled across ranks) used to compute val_wasserstein.",
    )

    parser.add_argument("--samples_per_combo", type=int, default=20000)
    parser.add_argument("--val_samples_per_combo", type=int, default=1000)

    # MFTR parameter ranges (JSON arrays like "[2.0, 10.0]")
    parser.add_argument("--combos", type=int, default=64)
    parser.add_argument("--val_combos", type=int, default=None)

    parser.add_argument("--m_range", type=str, default="[8.0, 8.0]")
    parser.add_argument("--mu_range", type=str, default="[7.0, 7.0]")
    parser.add_argument("--K_range", type=str, default="[8.0, 8.0]")
    parser.add_argument("--delta_range", type=str, default="[0.9, 0.9]")
    parser.add_argument("--omega_range", type=str, default="[2.0, 2.0]")

    parser.add_argument(
        "--normalize_conds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, normalize MFTR condition parameters to [-1, 1] using the training ranges.",
    )

    parser.add_argument(
        "--dist_type",
        type=str,
        default="amplitude",
        choices=["amplitude", "power"],
        help="Whether MFTR samples are amplitudes or powers",
    )

    # Evaluation (end-of-training)
    parser.add_argument(
        "--eval_max_mixture_samples",
        type=int,
        default=50000,
        help="Max number of mixture samples to use for end-of-training mixture evaluation",
    )
    parser.add_argument(
        "--eval_num_params_in",
        type=int,
        default=5,
        help="Number of in-range parameter sets to visualize/evaluate",
    )
    parser.add_argument(
        "--eval_num_params_out",
        type=int,
        default=5,
        help="Number of out-of-range parameter sets to visualize/evaluate",
    )
    parser.add_argument(
        "--eval_num_samples_per_param",
        type=int,
        default=5000,
        help="Number of samples to draw for each per-parameter evaluation",
    )

    parser.add_argument(
        "--eval_save_per_experiment_images",
        action="store_true",
        help=(
            "If set, also saves one image per conditional experiment under "
            "test_results/conditional_interpolation/ and test_results/conditional_extrapolation/"
        ),
    )

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()
