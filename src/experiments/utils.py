from datetime import datetime
from pathlib import Path

import click
import cv2
import numpy as np
import torch

from torch import Tensor, nn

from src.generate.generate_figures import generate_rectangle, generate_triangle
from src.models import load_model_from_checkpoint, model_for_name, BaseClassificationModel 


class TotalVarianceLoss(nn.Module):
    def __init__(self, kernel_size: int, channels: int) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(channels, channels, kernel_size, groups=channels, padding=(kernel_size - 1) // 2)
        self.conv.eval()
        for p in self.conv.parameters():
            p.requires_grad = False
        nn.init.zeros_(self.conv.bias)
        nn.init.ones_(self.conv.weight)
        self.conv.weight.data = self.conv.weight.data / (kernel_size ** 2)

    def __call__(self, x: Tensor) -> Tensor:
        a = self.conv(x)
        b = self.conv(x ** 2)
        return (b - a ** 2).mean(dim=[1, 2, 3])


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
        size: tuple[int, int, int],
        initialization: str,
        data_path: Path | None = Path('training-data-retrieval/data/histology_prior_images')
) -> Tensor:
    if initialization == 'random':
        return (0.5 + torch.randn(n_images, *size)).clip(0, 1)
    if initialization == 'random-2':
        return torch.randn(n_images, *size).clip(0, 1)
    if initialization == 'prior-inverted':
        data = [generate_rectangle() for _ in range(n_images // 2)] + [generate_triangle() for _ in range(n_images // 2)]
        data = np.stack(data, axis=0).astype('float32')
        return torch.from_numpy(data).permute(0, 3, 1, 2).contiguous()
    if initialization == 'prior-inverted-2':
        data = [generate_triangle() for _ in range(n_images // 2)] + [generate_rectangle() for _ in range(n_images // 2)]
        data = np.stack(data, axis=0).astype('float32')
        return torch.from_numpy(data).permute(0, 3, 1, 2).contiguous()

    if initialization == 'color':
        sample_images = 0.05 * torch.randn(n_images, 3, 128, 128)#.clip(0, 1)
        sample_images = sample_images + torch.tensor([179.0 / 255, 128.0 / 255, 147.0 / 255]).reshape(1, 3, 1, 1)
        return sample_images.clip(0, 1)

    assert data_path is not None

    result = []
    for p in data_path.glob('*.png'):
        image = cv2.imread(str(p))[::-1].astype('float32') / 255
        image = cv2.resize(image, (128, 128))
        result.append(image)
    
    if len(result) > n_images:
        result = result[:n_images]
    result = np.stack(result, axis=0)
    return torch.from_numpy(result).permute(0, 3, 1, 2)


def get_device(d: str) -> str:
    if d != 'auto':
        return d
    
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'
