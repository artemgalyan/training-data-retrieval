import typing as tp

from torch import nn, Tensor

from .base import BaseClassificationModel
from .blocks import ResidualBlock, ConvNormAct, Sequential


def activation_for_name(name: str) -> type:
    return {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
    }[name.lower()]


class ClassificationNet(BaseClassificationModel):
    def __init__(
        self,
        num_classes: int,
        configuration: list[tuple[int, int]],
        image_channels: int,
        dilation: int = 1,
        norm: type | str = nn.BatchNorm2d,
        activation: type | str = nn.ReLU,
        bias: bool = True
    ) -> None:
        """
        Residual network for image classification
        :param num_classes: number of classes for classification
        :param configuration: network configuration in format (dim, num_blocks)
        After sequence of blocks downsampling is performed
        """

        super().__init__(num_classes)

        if isinstance(activation, str):
            activation = activation_for_name(activation)
        
        if isinstance(norm, str):
            norm = nn.Identity if norm == 'none' else nn.BatchNorm2d
        
        self.save_hyperparameters()

        self.configuration = configuration
        self.image_channels = image_channels
        self.dilation = dilation
        self.norm = norm
        self.activation = activation
        self.bias = bias

        prev_dim = configuration[0][0]
        modules = [nn.Conv2d(image_channels, prev_dim, kernel_size=3, padding=1)]

        kw = dict(
            norm=norm,
            activation=activation,
            bias=bias,
            dilation=dilation
        )
        for dim, num_blocks in configuration:
            modules.append(
                ConvNormAct(prev_dim, dim, kernel_size=4, padding=1, stride=2, **kw)
            )
            modules.extend([
                ConvNormAct(dim, dim, kernel_size=3, padding=1, **kw)
                for _ in range(num_blocks)
            ])
            prev_dim = dim
        
        modules.append(nn.Conv2d(prev_dim, prev_dim, 3, padding=1, bias=bias))
        
        self.main = Sequential(*modules)
        self.classifier = nn.Linear(prev_dim, num_classes, bias=bias)
    
    def forward(
        self,
        x: Tensor,
        return_activations: bool = False
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        x, act = self.main(x)
        
        x = x.mean(dim=[-1, -2])
        x = self.classifier(x)
        if return_activations:
            return x, act
        return x
    
    def get_logits(self, samples: Tensor) -> Tensor:
        return self.forward(samples, return_activations=False)
    
    def dict_data(self) -> dict[str, tp.Any]:
        return {
            **super().dict_data(),
            'self.configuration': self.configuration,
            'image_channels': self.image_channels,
            'dilation': self.dilation,
            'norm': self.norm.__name__,
            'activation': self.activation.__name__,
            'bias': self.bias,
        }


class ClassificationResNet(BaseClassificationModel):
    def __init__(
        self,
        num_classes: int,
        configuration: list[tuple[int, int]],
        image_channels: int,
        dilation: int = 1,
        norm: type = nn.BatchNorm2d,
        activation: type = nn.ReLU,
        bias: bool = True
    ) -> None:
        """
        Residual network for image classification
        :param num_classes: number of classes for classification
        :param configuration: network configuration in format (dim, num_blocks)
        After sequence of blocks downsampling is performed
        """

        super().__init__(num_classes)

        if isinstance(activation, str):
            activation = activation_for_name(activation)

        self.save_hyperparameters()

        self.configuration = configuration
        self.image_channels = image_channels
        self.dilation = dilation
        self.norm = norm
        self.activation = activation
        self.bias = bias

        prev_dim = configuration[0][0]
        modules = [nn.Conv2d(image_channels, prev_dim, kernel_size=3, padding=1)]

        kw = dict(
            norm=norm,
            activation=activation,
            bias=bias,
            dilation=dilation
        )
        for dim, num_blocks in configuration:
            modules.append(
                nn.Conv2d(prev_dim, dim, kernel_size=4, padding=1, stride=2, bias=bias)
            )
            modules.extend([
                ResidualBlock(dim, dim, kernel_size=3, padding=1, **kw)
                for _ in range(num_blocks)
            ])
            prev_dim = dim
        
        self.main = Sequential(*modules)
        self.classifier = nn.Linear(prev_dim, num_classes)
    
    def forward(
        self,
        x: Tensor,
        return_activations: bool = False
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        x, act = self.main(x)
        
        x = x.mean(dim=[-1, -2])
        x = self.classifier(x)
        if return_activations:
            return x, act
        return x
        
    def get_logits(self, samples: Tensor) -> Tensor:
        return self.forward(samples, return_activations=False)
    
    def dict_data(self) -> dict[str, tp.Any]:
        return {
            **super().dict_data(),
            'self.configuration': self.configuration,
            'image_channels': self.image_channels,
            'dilation': self.dilation,
            'norm': self.norm.__name__,
            'activation': self.activation.__name__,
            'bias': self.bias,
        }
    