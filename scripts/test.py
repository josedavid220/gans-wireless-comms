from gans import GAN, WGAN_GP
import os
import os.path as path
from config import get_args
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset
import torch

args = get_args()

NUM_SAMPLES = args.num_test_samples
NUM_TEST_SAMPLES = args.num_test_samples
TESTS_SAVE_PATH = args.tests_save_path
PRECISION = 8
DEVICE = "cpu"
GAN_TYPE = args.gan_type
VERSION = args.version
DISTRIBUTION = args.distribution

version_dir = path.join("..", "logs", DISTRIBUTION, GAN_TYPE, f"version_{VERSION}")
ckpt_dir = path.join(version_dir, "checkpoints")

ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
if not ckpt_files:
    raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")

CKPT_PATH = path.join(ckpt_dir, ckpt_files[0])


def get_model():
    """Load model from checkpoint."""
    match GAN_TYPE:
        case "gan":
            model = GAN
        case "wgan_gp":
            model = WGAN_GP
        case _:
            raise ValueError(f"Unsupported GAN type: {GAN_TYPE}")

    return model.load_from_checkpoint(CKPT_PATH, num_test_samples=NUM_TEST_SAMPLES).to(
        DEVICE
    )


logger = TensorBoardLogger(save_dir="../logs", name=f"{DISTRIBUTION}/{GAN_TYPE}")
trainer = L.Trainer(accelerator="auto", logger=logger)
# Create dummy test dataloader just to prevent errors
dummy_data = torch.randn(1, 10)  # 1 sample, 10 features
dummy_dataset = TensorDataset(dummy_data)
dummy_dataloader = DataLoader(dummy_dataset, batch_size=1)

trainer.test(model=get_model(), dataloaders=dummy_dataloader)
