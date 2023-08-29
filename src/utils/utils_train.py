import torch
from torch import Tensor

def nanmean(v, nan_mask, inplace=True, **kwargs):
    """
    https://github.com/pytorch/pytorch/issues/21987#issuecomment-539402619
    """
    if not inplace:
        v = v.clone()
    v[nan_mask] = 0
    return v.sum(**kwargs) / (~nan_mask).float().sum(**kwargs)

def limit(tensor: Tensor, min=-5, max=10) -> Tensor:
    """
    Clamp tensor below specified limit. Useful for preventing unstable training when using logarithmic network outputs.
    """
    return torch.clamp(tensor, min=min, max=max)
