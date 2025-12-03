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
