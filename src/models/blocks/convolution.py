import torch

from torch import nn, Tensor

from .utils import LayerScale2d, DropPath


class ConvNormAct(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            padding: int,
            stride: int = 1,
            dilation: int = 1,
            norm: type = nn.BatchNorm2d,
            activation: type = nn.ReLU,
            bias: bool = True
     ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                bias=bias
            ),
            norm(out_channels),
            activation(),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)



class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            padding: int,
            stride: int = 1,
            dilation: int = 1,
            norm: type = nn.BatchNorm2d,
            activation: type = nn.ReLU,
            bias: bool = True
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                bias=bias
            ),
            norm(out_channels),
            activation(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            norm(out_channels),
            activation(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                bias=bias
            )
        )
    

    def forward(self, x: Tensor) -> Tensor:
        return x + self.blocks(x)


class ConvNextBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            drop_rate: float = 0.,
            layer_scale_init_value: float = 1e-6,
            stride: int = 1,
            output_dim: int | None = None,
            norm: type = nn.BatchNorm2d,
    ) -> None:
        super().__init__()

        if output_dim is None:
            output_dim = dim

        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=stride, groups=dim)
        self.norm = norm(dim)
        self.pointwise_conv1 = nn.Conv2d(dim, 4*dim, kernel_size=1)
        self.activation = nn.GELU()
        self.pointwise_conv2 = nn.Conv2d(4*dim, output_dim, kernel_size=1)
        self.layer_scale = LayerScale2d(output_dim, layer_scale_init_value) if layer_scale_init_value > 0 else nn.Identity()
        self.drop_path = DropPath(drop_rate) if drop_rate is not None and drop_rate > 0. else nn.Identity()

        if stride == 1 and dim != output_dim:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(dim, output_dim, kernel_size=1, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        z = x
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.pointwise_conv1(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.layer_scale(x)
        x = self.shortcut(z) + self.drop_path(x)
        return x
