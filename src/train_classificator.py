import click
import json

from pathlib import Path

import lightning as L

import torchvision.transforms.v2 as T

from lightning.pytorch.callbacks import ModelCheckpoint, ProgressBar, StochasticWeightAveraging, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from .datasets import ClassificationDataset, HistologyDataset
from .models import model_for_name, BaseClassificationModel 


def get_transforms(
        image_size: tuple[int, int], 
        noise_params: dict | None
    ) -> tuple:
    train_transforms = T.Compose([
        T.ToTensor(),
        lambda x: x if noise_params is None else T.GaussianNoise(**noise_params),
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=15),
        T.RandomResizedCrop(size=image_size)
    ])

    test_transforms = T.Compose([
        T.ToTensor(),
        T.Resize(size=image_size)
    ])
    return train_transforms, test_transforms


def load_model(config: dict) -> BaseClassificationModel:
    model_type = model_for_name(config['model_type'])
    return model_type(
        *config.get('args', tuple()),
        **config.get('kwargs', {})
    )

def load_datasets(config: dict) -> tuple[ClassificationDataset, ClassificationDataset]:
    train_t, test_t = get_transforms(tuple(config['image_size']), config.get('noise', None))
    train = config['train']
    val = config['val']
    return HistologyDataset(**train, transforms=train_t), HistologyDataset(**val, transforms=test_t)


@click.command()
@click.argument('run_configuration', type=click.Path(exists=True), required=True, help='Path to .json run configuration file')
def main(run_configuration: str) -> None:
    config_path = Path(run_configuration)
    if not config_path.exists() or not config_path.is_file():
        click.echo('Run configuration file should exist!', error=True)
        return
    
    with open(run_configuration, 'r') as file:
        config = json.loads(file.read())
    
    model = load_model(config['model'])
    train_dataset, val_dataset = load_datasets(config['datasets'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    trainer = L.Trainer(
      **config['trainer'],
      logger=WandbLogger(
          project='Training data retrieval',
          name=config['run_name']
      ),
      check_val_every_n_epoch=5,
      callbacks=[
          ModelCheckpoint(
              monitor='val_loss',
              dirpath=config.get('checkpoint_dir', 'models'),
              filename=config.get('saving_format', '{epoch}-{val_accuracy:.2f}'),
              save_top_k=3,
              save_last=True
          ),
          ProgressBar(),
          EarlyStopping()
      ]
    )
    click.echo('Starting the run')
    trainer.fit(
        model=model,
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
    )
    click.echo('Run finished')
