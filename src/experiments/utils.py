import datetime

from pathlib import Path

import click
import cv2
import numpy as np
import torch

from torch import Tensor

from src.models import load_model_from_checkpoint, model_for_name, BaseClassificationModel 


def log(message: str) -> None:
    """
    Logs message into stdout
    """
    click.echo(f'[{datetime.now():%H:%M:%S}] {message}')


def load_model(config: dict) -> BaseClassificationModel:
    """
    Loads model from "model" part of config
    """
    model_type = model_for_name(config['model_type'])
    return load_model_from_checkpoint(
        config['checkpoint'],
        model_type
    )


def save_data(images: Tensor, save_dir: Path) -> None:
    images = images.clone().cpu().detach().numpy()

    if not save_dir.exists():
        save_dir.mkdir()

    for i in range(images.shape[0]):
        image = images[i].transpose(1, 2, 0)[..., ::-1]
        save_path = save_dir / f'{i}.png'
        cv2.imwrite(
            str(save_path),
            (255 * image).astype('uint8')
        )


def initialize_sample_images(
        n_images: int,
        initialization: str,
        data_path: Path | None = Path('data/histology_prior_images')
) -> Tensor:
    assert initialization in ['random', 'color', 'prior']

    if initialization == 'random':
        return (0.5 + torch.randn(n_images, 3, 128, 128)).clip(0, 1)
    if initialization == 'color':
        sample_images = 0.05 * torch.randn(n_images, 3, 128, 128)#.clip(0, 1)
        sample_images = sample_images + torch.tensor([179.0 / 255, 128.0 / 255, 147.0 / 255]).reshape(1, 3, 1, 1)
        return sample_images.clip(0, 1)

    assert data_path is not None

    result = []
    for p in data_path.glob('*.png'):
        image = cv2.imread(str(p))[::-1].astype('float32') / 255
        result.append(image)
    
    if len(result) > n_images:
        result = result[:n_images]
    result = np.stack(result, axis=0)
    return torch.from_numpy(result)
