import json

from collections import defaultdict
from pathlib import Path

import click
import pandas as pd
import torch
import torch.nn.functional as F

from torch.autograd import grad
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm

from src.experiments import log, load_model, save_data, initialize_sample_images, get_device, TotalVarianceLoss
from src.utils import insert_listener, BatchNorm2dListener


class Accumulator:
    """
    Used for batchnorm losses
    """
    def __init__(self) -> None:
        self.losses = []
    
    def __call__(self, item: torch.Tensor) -> None:
        self.losses.append(item)
    
    def get_loss(self) -> None:
        loss = torch.cat(self.losses)
        self.losses = []
        return loss
    

@click.command()
@click.argument('run_configuration', type=click.Path(exists=True), required=True)
@click.argument('n_images', type=int, required=True)
@click.argument('n_iterations', type=int, required=True)
@click.argument('val_every', type=int, required=True)
@click.argument('initialization', type=str, required=True)
@click.argument('alpha', type=float, required=True)
@click.option('-g', '--gamma', type=float, default=5)
@click.option('-t', '--theta', type=float, default=1)
@click.option('-p', type=float, default=2.0, help='Grad penalty norm')
@click.option('-sg', '--save_to_gdrive', type=bool, default=False, help='Whether to save to google drive')
@click.option('-m', '--mode', type=str, default='train', help='NN mode to run: train or eval')
def main(
    run_configuration: str,
    n_images: int,
    n_iterations: int,
    val_every: int,
    initialization: str,
    alpha: float,
    gamma: float,
    theta: float,
    p: int,
    save_to_gdrive: bool,
    mode: str
) -> None:
    config_path = Path(run_configuration)
    if not config_path.exists() or not config_path.is_file():
        click.echo('Run configuration file should exist!', error=True)
        return
    
    with open(run_configuration, 'r') as file:
        config = json.loads(file.read())

    log('Successfully loaded the configuration')

    model = load_model(config['model'])
    
    model = load_model(config['model'])
    if mode == 'train':
        model.train()
    else:
        model.eval()
    log('Successfully loaded model')

    mean_accumulator, var_accumulator = Accumulator(), Accumulator()
    insert_listener(model, lambda bn: BatchNorm2dListener(bn, lambda x, y: torch.abs(x - y) ** p, mean_accumulator, var_accumulator))
    log('Inserted BatchNorm2dListeners instead of BatchNorm2ds')

    device = get_device(config.get('device', 'cpu'))
    log(f'Using {device}')
    model.to(device)

    smooth = 'smooth' if config['model'].get('activation', 'relu').lower() != 'relu' else 'non-smooth'
    run_name = f'{config["model"]["model_type"]}-zero-grad-{n_images}-{n_iterations}-{initialization}-{smooth}-alpha-{alpha}-p-{p}'
    if save_to_gdrive:
        save_path = Path('/content/drive/MyDrive/experiments') / run_name
    else:
        save_path = Path(run_name)
    if not save_path.exists():
        save_path.mkdir()

    sample_images = initialize_sample_images(n_images, config.get('size', (1, 128, 128)), initialization)
    sample_images = sample_images.to(device)
    sample_images = sample_images.clip(0, 1)
    sample_images.requires_grad = True

    tv = TotalVarianceLoss(kernel_size=5, channels=1)

    losses = defaultdict(list)
    optim = SGD([sample_images], lr=0.1)
    scheduler = StepLR(optim, 10_000, gamma=0.1)
    target = torch.arange(n_images, dtype=torch.long).to(device) % 10

    bar = tqdm(range(n_iterations))
    save_data(sample_images, save_path / 'initial')
    for i in bar:
        optim.zero_grad()
        model.zero_grad()
        y = model(sample_images)
        y = F.cross_entropy(y, target)
        y_grad = grad(
            y.sum(),
            model.parameters(),
            create_graph=True
        )
        
        y_grad = torch.cat([g.reshape(-1) for g in y_grad])
        grad_loss = sum(torch.abs(z ** p).mean() for z in y_grad)
        bn_mean_loss = mean_accumulator.get_loss().sum()
        bn_var_loss = var_accumulator.get_loss().sum()
        tv_loss = tv(sample_images).mean()
        loss = grad_loss + alpha * (bn_mean_loss + bn_var_loss) + gamma * tv_loss + theta * y.mean()
        loss.backward()
        optim.step()
        model.zero_grad()
        losses['Gradient loss'].append(float(grad_loss.cpu().item()))
        losses['BN mean loss'].append(float(bn_mean_loss.cpu().item()))
        losses['BN var loss'].append(float(bn_var_loss.cpu().item()))
        losses['TV loss'].append(float(tv_loss.cpu().item()))
        losses['Class loss'].append(float(y.mean().cpu().item()))

        bar.set_description(f'{float(loss.cpu().item())}, {float(bn_mean_loss.cpu().item())} {float(bn_var_loss.cpu().item())}')
        scheduler.step()
        losses['Loss'].append(float(loss.cpu().item()))

        with torch.no_grad():
            sample_images.clip_(0, 1)

        if i % val_every == 0:
            save_data(sample_images, save_path / f'epoch-{i}')
        
    losses = pd.DataFrame(losses)
    losses.to_csv(str(save_path / 'losses.csv'))


if __name__ == '__main__':
    main()
