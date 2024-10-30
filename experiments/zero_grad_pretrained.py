import click
import json

from datetime import datetime
from pathlib import Path

import cv2
import torch

from torch.autograd import grad
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet18, ResNet18_Weights
from tqdm.auto import tqdm


def log(message: str) -> None:
    click.echo(f'[{datetime.now():%H:%M:%S}] {message}')


def save_data(images: torch.Tensor, save_dir: Path) -> None:
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


@click.command()
@click.argument('n_images', type=int, required=True)
@click.argument('n_iterations', type=int, required=True)
@click.argument('val_every', type=int, required=True)
@click.argument('initialization', type=str, required=True)
def main(
    run_configuration: str,
    n_images: int,
    n_iterations: int,
    val_every: int,
    initialization: str
) -> None:
    config_path = Path(run_configuration)
    if not config_path.exists() or not config_path.is_file():
        click.echo('Run configuration file should exist!', error=True)
        return
    
    with open(run_configuration, 'r') as file:
        config = json.loads(file.read())

    log('Successfully loaded the configuration')

    model = resnet18(ResNet18_Weights.IMAGENET1K_V1)
    log('Successfully loaded model')

    device = config.get('device', 'cpu')
    log(f'Using {device}')
    model.to(device)

    run_name = f'{config["model"]["model_type"]}-zero-grad-{n_images}-{n_iterations}-{initialization}'
    save_path = Path(run_name)
    if not save_path.exists():
        save_path.mkdir()

    if initialization == 'random':
        log('Using random initialization')
        sample_images = 0.5 * torch.randn(n_images, 3, 224, 224)
    elif initialization == 'color':
        log('Using color initialization')
        sample_images = 0.05 * torch.randn(n_images, 3, 224, 224)
        sample_images = sample_images + torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    
    sample_images = sample_images.to(device)
    sample_images = sample_images.clip(0, 1)
    sample_images.requires_grad = True

    losses = []
    optim = SGD([sample_images], lr=0.1)
    scheduler = StepLR(optim, 10_000, gamma=0.1)
    target = torch.ones((n_images,), dtype=torch.float32).to(device)
    target[n_images // 2:] = 0

    bar = tqdm(range(n_iterations))
    save_data(sample_images, save_path / 'initial')
    for i in bar:
        optim.zero_grad()
        model.zero_grad()
        y = model(sample_images)
        # y = F.binary_cross_entropy_with_logits(y[:, 1].reshape(-1), target)
        y_grad = grad(
            y.sum(),
            sample_images,
            create_graph=True
        )[0]
        
        loss = (y_grad ** 2).sum()
        loss.backward()
        optim.step()
        model.zero_grad()
        bar.set_description(str(loss.cpu().item()))
        scheduler.step()
        losses.append(float(loss.cpu().item()))

        with torch.no_grad():
            sample_images.clip_(0, 1)

        if i % val_every == 0:
            save_data(sample_images, save_path / f'epoch-{i}')
        
    with open(str(save_path / 'losses.txt'), 'w') as file:
        file.writelines(map(lambda x: f'{x}\n', losses))


if __name__ == '__main__':
    main()
