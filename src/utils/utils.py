from collections.abc import Callable

from torch import nn, Tensor



def get_params_vector(model: nn.Module) -> Tensor:
    """
    Returns model parameters as vector
    """
    return nn.utils.parameters_to_vector(model.parameters())


class BatchNorm2dListener(nn.Module):
    """
    This class is a wrapper for nn.BatchNorm2d
    The goal is to get difference between mean and std of batch and ones
    stored in wrapped BN
    """

    def __init__(
        self,
        wrapped: nn.BatchNorm2d,
        metric: Callable[[Tensor, Tensor], Tensor],
        mean_callback: Callable[[Tensor], None],
        var_callback: Callable[[Tensor], None]
    ) -> None:
        super().__init__()

        self.wrapped = wrapped
        self.metric = metric
        self.mean_callback = mean_callback
        self.var_callback = var_callback

        self.target_mean = self.wrapped.running_mean.detach().clone()
        self.target_var = self.wrapped.running_var.detach().clone()

    def forward(self, x):
        mean = x.mean(axis=[0, 2, 3])
        var = x.var(axis=[0, 2, 3])

        mean_distance = self.metric(mean, self.target_mean)
        var_distance = self.metric(var, self.target_var)

        self.mean_callback(mean_distance)
        self.var_callback(var_distance)

        return self.wrapped(x)


def insert_listener(
    model: nn.Module,
    listener_factory: Callable[[nn.BatchNorm2d], BatchNorm2dListener]
) -> None:
    """
    Performs model surgery, replacing all BatchNorm2d modules with listeners
    """

    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(model, name, listener_factory(child))
        else:
            insert_listener(child, listener_factory)
