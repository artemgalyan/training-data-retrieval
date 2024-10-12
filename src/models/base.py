import typing as tp

from abc import abstractmethod

import lightning as L
import torch

from torch import nn, Tensor
from torch.optim import AdamW


class BaseModel(L.LightningModule):
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @abstractmethod
    def dict_data(self) -> dict[str, tp.Any]:
        pass


class BaseClassificationModel(BaseModel):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.num_classes = num_classes
        if self.num_classes == 2:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

    def dict_data(self) -> dict[str, tp.Any]:
        return {
            'num_classes': self.num_classes
        }

    def training_step(self, batch: Tensor, *args: tp.Any) -> Tensor:
        x, y = batch
        z = self.get_logits(x)
        loss = self.loss(z, y)

        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self.get_logits(x)
        if self.num_classes == 2:
            y_hat = y_hat[:, 1:]
            y = y.float()
        loss = self.loss(y_hat, y.view(-1, 1))
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self.get_logits(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss')
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        return optimizer
    
    @abstractmethod
    def get_logits(self, samples: Tensor) -> Tensor:
        pass