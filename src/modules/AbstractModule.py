import torch

from utilities.DelayedFunctions import DelayedFunctions

class AbstractModule(torch.nn.Module):
    _delayed : DelayedFunctions

    def __init__(self):
        super().__init__()
        self._delayed = DelayedFunctions()

    def forward(self, data:torch.Tensor):
        raise NotImplementedError