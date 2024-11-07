from pathlib import Path

import torch

from .base import BaseModel


def load_model_from_checkpoint(checkpoint_path: Path | str, model_type: type, map_location=torch.device('cpu')) -> BaseModel:
    state_dict = torch.load(str(checkpoint_path), map_location=map_location)
    model = model_type(**state_dict['hyper_parameters'])
    model.load_state_dict(state_dict['state_dict'])
    return model.eval()
