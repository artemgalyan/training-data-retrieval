import json

from pathlib import Path

import click
import pandas as pd
import torch
import torch.nn.functional as F

from torch.autograd import grad
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm

from src.experiments import log, load_model, save_data, initialize_sample_images


@click.command()
@click.argument('run_configuration', type=click.Path(exists=True), required=True)
@click.argument('n_images', type=int, required=True)
@click.argument('n_iterations', type=int, required=True)
@click.argument('val_every', type=int, required=True)
@click.argument('initialization', type=str, required=True)
@click.option('-p', type=float, default=2.0, help='Grad penalty norm')
@click.option('-sg', '--save_to_gdrive', type=bool, default=False, help='Whether to save to google drive')
def main(
    run_configuration: str,
    n_images: int,
    n_iterations: int,
    val_every: int,
    initialization: str,
    p: int,
    save_to_gdrive: bool
) -> None:
    config_path = Path(run_configuration)
    if not config_path.exists() or not config_path.is_file():
        click.echo('Run configuration file should exist!', error=True)
        return
    
    with open(run_configuration, 'r') as file:
        config = json.loads(file.read())

    log('Successfully loaded the configuration')

    model = load_model(config['model']).eval()
    log('Successfully loaded model')

    device = config.get('device', 'cpu')
    log(f'Using {device}')
    model.to(device)

    smooth = 'smooth' if config['model'].get('activation', 'relu').lower() != 'relu' else 'non-smooth'
    run_name = f'{config["model"]["model_type"]}-zero-grad-{n_images}-{n_iterations}-{initialization}-{smooth}-p-{p}'
    if save_to_gdrive:
        save_path = Path('/content/drive/MyDrive/experiments') / run_name
    else:
        save_path = Path(run_name)
    if not save_path.exists():
        save_path.mkdir()

    sample_images = initialize_sample_images(n_images, initialization)
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
        y = F.binary_cross_entropy_with_logits(y[:, 1].reshape(-1), target)
        y_grad = grad(
            y.sum(),
            model.parameters(),
            create_graph=True
        )
        
        y_grad = torch.cat([g.view(-1) for g in y_grad])
        loss = sum(torch.abs(y ** p).mean() for y in y_grad)
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
        
    
    losses = pd.DataFrame({
        'Loss': losses
    })
    losses.to_csv(str(save_path / 'losses.csv'))

    with open(str(save_path / 'losses.txt'), 'w') as file:
        file.writelines(map(lambda x: f'{x}\n', losses))


if __name__ == '__main__':
    main()
