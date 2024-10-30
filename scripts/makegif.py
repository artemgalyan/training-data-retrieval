from pathlib import Path

import click
import imageio


@click.command()
@click.argument('experiment_path', type=click.Path(exists=True))
def main(experiment_path: Path) -> None:
    ep = Path(experiment_path)
    dirs = ep.glob('*')
    dirs = [d for d in dirs if d.name.startswith('epoch')]
    files = [d.name for d in dirs[0].glob('*.png')]
    output_dir = ep / 'gif'
    if not output_dir.exists():
        output_dir.mkdir()

    for file in files:
        image = []
        for d in dirs:
            image.append(imageio.imread(str(d / file)))
        
        name = file[:-3] + 'gif'
        print(name)
        imageio.mimsave(output_dir / name, image, duration=0.5)

main()
