from gans import GAN, WGAN_GP
from local_datasets import RayleighDataset, NakagamiDataset
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from config import get_args
import json

args = get_args()

DISTRIBUTION = args.distribution
GAN_TYPE = args.gan_type
LATENT_DIM = args.latent_dim
NUM_SAMPLES = args.num_samples
NUM_SAMPLES_VAL = args.num_samples // 10
NUM_TEST_SAMPLES = args.num_test_samples
BATCH_SIZE = args.batch_size
G_EVERY_K_STEPS = args.g_every_k_steps
MAX_EPOCHS = args.epochs
NUM_WORKERS = args.num_workers

distribution_params = {}
if args.distribution_params:
    try:
        distribution_params = json.loads(args.distribution_params)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format for --distribution_params argument.")
else:
    print("No --distribution_params argument provided.")


def get_model():
    # Parameters shared by all models
    model_params = {
        "latent_dim": LATENT_DIM,
        "g_every_k_steps": G_EVERY_K_STEPS,
        "distribution_name": DISTRIBUTION,
        "distribution_params": distribution_params,
        "num_test_samples": NUM_TEST_SAMPLES,
    }

    match GAN_TYPE:
        case "gan":
            model = GAN
        case "wgan_gp":
            model_params["lambda_gp"] = args.lambda_gp
            model = WGAN_GP
        case _:
            raise ValueError(f"Unsupported GAN type: {GAN_TYPE}")

    return model(**model_params)


def get_dataset(num_samples, seed: int | None):
    dataset_params = {
        "num_samples": num_samples,
        **distribution_params,
    }

    match DISTRIBUTION:
        case "rayleigh":
            dataset = RayleighDataset
        case "nakagami":
            dataset = NakagamiDataset
        case _:
            raise ValueError(f"Unsupported distribution: {DISTRIBUTION}")

    return dataset(**dataset_params)


def get_dataloader(dataset, batch_size):
    return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=NUM_WORKERS)


model = get_model()

train_dataset = get_dataset(num_samples=NUM_SAMPLES, seed=42)
train_dataloader = get_dataloader(train_dataset, batch_size=BATCH_SIZE)

val_dataset = get_dataset(num_samples=NUM_SAMPLES_VAL, seed=0)
val_dataloader = get_dataloader(val_dataset, batch_size=NUM_SAMPLES_VAL)

logger = TensorBoardLogger(save_dir="../logs", name=f"{DISTRIBUTION}/{GAN_TYPE}")
trainer = L.Trainer(max_epochs=MAX_EPOCHS, accelerator="auto", logger=logger)
trainer.fit(
    model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)

# Run tests after training
print("Running goodness-of-fit tests...")
# Maybe will need to add a custom test dataloader in the future
# for empiric distributions. For now, it's just there to avoid errors.
trainer.test(ckpt_path="best", dataloaders=val_dataloader)
