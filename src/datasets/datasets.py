from glob import glob
from pathlib import Path

import cv2
import numpy as np

from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


def dataset_from_name(name: str) -> type:
    return {
        'HistologyDataset': HistologyDataset,
        'FigureDataset': FigureDataset,
        'MNISTDataset': MNISTDataset,
        'BinaryMNISTDataset': BinaryMNISTDataset,
        'TinyBinaryMNISTDataset': TinyBinaryMNISTDataset,
        'SmallBinaryMNISTDataset': SmallBinaryMNISTDataset
    }[name]


def read_image(path: Path | str) -> NDArray:
    return cv2.imread(str(path))[..., ::-1].astype('float32') / 255


class ClassificationDataset(Dataset):
    """
    Class representing all classification datasets
    """
    def __init__(
            self,
            classes: list[str]
        ) -> None:
        self.classes = classes
        self.class_to_index = {c: i for i, c in enumerate(sorted(classes))}
    
    @property
    def num_classes(self) -> int:
        return len(self.classes)


class HistologyDataset(ClassificationDataset):
    def __init__(
        self, split_path: Path | str, transforms, classes: list[str], keep_in_memory: bool = False
    ) -> None:
        super().__init__(classes)

        self.split_path = Path(split_path)
        self.transforms = transforms
        self.classes = classes
        self.keep_in_memory = keep_in_memory

        self.image_files = []
        self.labels = []

        for clazz in self.classes:
            files = list(glob(f'{self.split_path / clazz}/*.png'))
            self.image_files.extend(files)
            self.labels.extend([clazz] * len(files))
        
        if keep_in_memory:
            self.preloaded_images = list(map(read_image, self.image_files))
        else:
            self.preloaded_images = []
    
    def __len__(self) -> None:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        if self.keep_in_memory:
            image = self.preloaded_images[index]
        else:
            image = read_image(self.image_files[index])
        label = self.class_to_index[self.labels[index]]
        return self.transforms(image), label
        

class FigureDataset(ClassificationDataset):
    def __init__(
            self, split_path: Path | str, transforms, classes: list[str], keep_in_memory: bool = True
    ) -> None:
        super().__init__(classes)
        
        self.split_path = Path(split_path)
        self.transforms = transforms
        self.classes = classes
        self.keep_in_memory = keep_in_memory

        self.image_files = []
        self.labels = []

        for clazz in self.classes:
            files = list(glob(f'{self.split_path / clazz}/*.png'))
            self.image_files.extend(files)
            self.labels.extend([clazz] * len(files))
        
        if keep_in_memory:
            self.preloaded_images = [read_image(f)[..., :1] for f in self.image_files]
        else:
            self.preloaded_images = []
    
    def __len__(self) -> None:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        if self.keep_in_memory:
            image = self.preloaded_images[index]
        else:
            image = read_image(self.image_files[index])[..., :1]
        
        label = self.class_to_index[self.labels[index]]
        return self.transforms(image), label
    

class MNISTDataset(ClassificationDataset):
    def __init__(
            self, split_path: Path | str, transforms, classes: list[str], **kw
    ) -> None:
        super().__init__(classes)

        is_train = str(split_path).endswith('train')
        self.dataset = MNIST(
            'mnist-dataset-' + ('train' if is_train else 'val'), 
            train=is_train,
            transform=transforms,
            download=True
        )

        self.split_path = Path(split_path)
        self.transforms = transforms
        self.classes = classes

    def __len__(self) -> None:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self.dataset[index]


class BinaryMNISTDataset(ClassificationDataset):
    def __init__(
            self, split_path: Path | str, transforms, classes: list[str], **kw
    ) -> None:
        super().__init__(classes)

        self.dataset = MNISTDataset(split_path, transforms, classes, **kw)

    def __len__(self) -> None:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image, label = self.dataset[index]
        return image, label % 2


class TinyBinaryMNISTDataset(ClassificationDataset):
    def __init__(
            self, split_path: Path | str, transforms, classes: list[str], **kw
    ) -> None:
        super().__init__(classes)

        self.transforms = transforms
        ds = MNISTDataset(split_path, transforms=None, classes=classes, **kw)
        self.dataset = [ds[i] for i in range(500)]

    def __len__(self) -> None:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image, label = self.dataset[index]
        return self.transforms(image), label % 2


class SmallBinaryMNISTDataset(ClassificationDataset):
    def __init__(
            self, split_path: Path | str, transforms, classes: list[str], **kw
    ) -> None:
        super().__init__(classes)

        self.transforms = transforms
        ds = MNISTDataset(split_path, transforms=None, classes=classes, **kw)
        self.dataset = [ds[i] for i in range(1000)]

    def __len__(self) -> None:
        return 1000

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image, label = self.dataset[index]
        return self.transforms(image), label % 2
