import torch

from ..AbstractModule import AbstractModule

class AbstractCompressor(AbstractModule):
    def __init__(self):
        super().__init__()

    def compress(self, data:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decompress(self, data:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, data:torch.Tensor):
        return self.decompress(self.decompress(data))
