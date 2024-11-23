import click
import json

from datetime import datetime
from pathlib import Path

import lightning as L
import torchvision.transforms.v2 as T

from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.datasets import ClassificationDataset, HistologyDataset, FigureDataset
from src.models import model_for_name, BaseClassificationModel 


def log(message: str) -> None:
    click.echo(f'[{datetime.now():%H:%M:%S}] {message}')


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
    if config.get('dataset') == 'FigureDataset':
        dataset = FigureDataset
    else:
        dataset = HistologyDataset

    return dataset(**train, transforms=train_t), dataset(**val, transforms=test_t)


@click.command()
@click.argument('run_configuration', type=click.Path(exists=True), required=True)
@click.option('-g', '--save_to_gdrive', type=bool, default=False)
def main(run_configuration: str, save_to_gdrive: bool) -> None:
    config_path = Path(run_configuration)
    if not config_path.exists() or not config_path.is_file():
        click.echo('Run configuration file should exist!', error=True)
        return
    
    with open(run_configuration, 'r') as file:
        config = json.loads(file.read())

    log('Successfully loaded the configuration')

    model = load_model(config['model'])
    log('Successfully loaded model')
    train_dataset, val_dataset = load_datasets(config['datasets'])

    num_workers = config['datasets'].get('num_workers', 0)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=num_workers)

    if save_to_gdrive:
        saving = Path('/content/gdrive/MyDrive/') / config.get('checkpoint_dir', 'models') / config['run_name']
    else:
        saving = 'checkpoint_dir', 'models'
    trainer = L.Trainer(
      **config['trainer'],
      logger=WandbLogger(
          project='Training data retrieval',
          name=config['run_name']
      ),
      check_val_every_n_epoch=1,
      callbacks=[
          ModelCheckpoint(
              monitor='val_loss',
              dirpath=str(saving),
              filename=config.get('saving_format', '{epoch}-{val_accuracy:.2f}'),
              save_top_k=3,
              save_last=True
          ),
          TQDMProgressBar(),
          LearningRateMonitor()
      ]
    )
    click.echo('Starting the run')
    trainer.fit(
        model=model,
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
    )
    click.echo('Run finished')
