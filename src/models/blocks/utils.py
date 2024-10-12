import torch

from torch import Tensor, nn


class Sequential(nn.Module):
    def __init__(self, *args: nn.Module) -> None:
        super().__init__()

        self.main = nn.ModuleList(list(args))
    
    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, list[Tensor]]:
        activations = []
        for m in self.main:
            x = m(x)
            activations.append(x)
        return x, activations


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)


class LayerScale2d(nn.Module):
    def __init__(self, dim: int, layer_scale_init_value: float):
        super().__init__()
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)
        
    def forward(self, x: Tensor) -> Tensor:
        x = x * self.gamma
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    @property
    def _device(self):
        return next(self.parameters()).device

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = (torch.rand(*shape) < keep_prob).to(dtype=x.dtype, device=self._device)
        output = (x * random_tensor) / keep_prob
        return output
