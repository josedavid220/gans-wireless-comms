from gans import GAN
from local_datasets.rayleigh_dataset import RayleighDataset
from torch.utils.data import DataLoader
import lightning as L
from config import get_args

args = get_args()

NUM_SAMPLES = args.num_samples
NUM_SAMPLES_VAL = args.num_samples // 10
BATCH_SIZE = args.batch_size
G_EVERY_K_STEPS = args.g_every_k_steps
MAX_EPOCHS = args.epochs
NUM_WORKERS = args.num_workers

model = GAN(g_every_k_steps=G_EVERY_K_STEPS)

train_dataset = RayleighDataset(num_samples=NUM_SAMPLES)
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
)
val_dataset = RayleighDataset(num_samples=NUM_SAMPLES_VAL)
val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=NUM_SAMPLES_VAL, num_workers=NUM_WORKERS
)

trainer = L.Trainer(
    max_epochs=MAX_EPOCHS, accelerator="auto", default_root_dir="../logs"
)
trainer.fit(
    model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)
