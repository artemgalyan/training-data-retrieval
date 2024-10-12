from .base import *
from .classification import * 


def model_for_name(clazz: str) -> type[BaseModel]:
    """
    Returns model clazz from name
    """
    mapping = {
        'ClassificationNet': ClassificationNet,
        'ClassificationResNet': ClassificationResNet
    }

    if clazz not in mapping:
        raise ValueError(f'{clazz} - unknown model class file!')
    
    return mapping[clazz]
