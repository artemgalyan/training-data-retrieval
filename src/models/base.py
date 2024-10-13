import typing as tp

from abc import abstractmethod

import lightning as L
import torch

from torch import nn, Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR


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
        if self.num_classes == 2:
            z = z[:, 1]
            y = y.float()
        
        loss = self.loss(z, y)

        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        x, y = batch
        z = self.get_logits(x)
        if self.num_classes == 2:
            z = z[:, 1]
            y = y.float()
        
        loss = self.loss(z, y)

        if self.num_classes == 2:
            labels = z > 0
        else:
            labels = z.argmax(axis=1)
        
        accuracy = (labels == y).float().mean()
        self.log_dict({
            'val_loss': loss,
            'val_accuracy': float(accuracy.cpu().item())
        })


    def test_step(self, batch: Tensor, batch_idx: int) -> None:
        x, y = batch
        z = self.get_logits(x)
        if self.num_classes == 2:
            z = z[:, 1]
            y = y.float()
        
        loss = self.loss(z, y)

        if self.num_classes == 2:
            labels = z > 0
        else:
            labels = z.argmax(axis=1)
        
        accuracy = (labels == y).float().mean()
        self.log_dict({
            'test_loss': loss,
            'test_accuracy': float(accuracy.cpu().item())
        })

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.8)
        return [optimizer], [scheduler]
    
    @abstractmethod
    def get_logits(self, samples: Tensor) -> Tensor:
        pass