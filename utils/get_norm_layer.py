import functools
from torch.functional import norm

import torch.nn as nn


def get_norm_layer(norm_type: str) -> nn:
    if norm_type == "batch":
        norm_layer =  functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError(f"norm_type {norm_type} is not implemented.")
    
    return norm_layer
