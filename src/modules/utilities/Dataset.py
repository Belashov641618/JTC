from __future__ import annotations

import torch
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Optional, Literal, Union

from ..utilities.DelayedFunctions import DelayedFunctions
from ..AbstractModule import AbstractModule

# Constants
root:Optional[str] = os.path.join(os.path.expanduser("~" + os.getlogin()), "Documents", "PyTorch Datasets")

LiteralDataSets = Literal['MNIST', 'Flowers']

class Dataset:
    _delayed : DelayedFunctions

    _train : DataLoader
    _test  : DataLoader
    def _reload(self):
        if self._dataset == 'MNIST':
            transformation = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.Resize((self._width, self._height), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.ToTensor(),
                    transforms.ConvertImageDtype(self._dtype)
                ])
            self._train = DataLoader(datasets.MNIST(root=root, train=True, transform=transformation, download=True), batch_size=self._batch, shuffle=True)
            self._test  = DataLoader(datasets.MNIST(root=root, train=False, transform=transformation, download=True), batch_size=self._batch, shuffle=True)
        elif self._dataset == 'Flowers':
            transformation = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((self._width, self._height), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(self._dtype)
            ])
            self._train = DataLoader(datasets.Flowers102(root=root, split='train', transform=transformation, download=True), batch_size=self._batch, shuffle=True)
            self._test  = DataLoader(datasets.Flowers102(root=root, split='test',  transform=transformation, download=True), batch_size=self._batch, shuffle=True)
        else: raise ValueError
    @property
    def train(self):
        self._delayed.launch()
        return self._train
    @property
    def test(self):
        self._delayed.launch()
        return self._test

    # Properties
    _dataset : LiteralDataSets
    @property
    def dataset(self):
        class DatasetSelector:
            _self : Dataset
            def __init__(self, _self:Dataset): self._self = _self
            def get(self): return self._self._dataset

            def set(self, dataset:LiteralDataSets):
                if self._self._dataset != dataset:
                    self._self._delayed.add(self._self._reload)
                self._self._dataset = dataset
            def __eq__(self, dataset:Union[LiteralDataSets,DatasetSelector]):
                if isinstance(dataset, DatasetSelector):
                    super().__eq__(dataset)
                else: self.set(dataset)

            def mnist(self):    self.set('MNIST')
            def flowers(self):  self.set('Flowers')
        return DatasetSelector(self)

    _batch : int
    @property
    def batch(self):
        return self._batch
    @batch.setter
    def batch(self, size:int):
        if size != self._batch:
            self._delayed.add(self._reload)
            self._batch = size

    _width : int
    @property
    def width(self):
        return self._width
    @width.setter
    def width(self, pixels:int):
        if pixels != self._width:
            self._delayed.add(self._reload)
            self._width = pixels

    _height : int
    @property
    def height(self):
        return self._height
    @height.setter
    def height(self, pixels:int):
        if pixels != self._height:
            self._delayed.add(self._reload)
            self._height = pixels

    _dtype : torch.dtype
    @property
    def dtype(self):
        return self._dtype
    @dtype.setter
    def dtype(self, type:torch.dtype):
        if type != self._dtype:
            self._delayed.add(self._reload)
            self._dtype = type

    def __init__(self, dataset:LiteralDataSets=None, batch:int=None, width:int=None, height:int=None, dtype:torch.dtype=torch.float32):
        self._delayed = DelayedFunctions()

        if dataset is not None: self.dataset.set(dataset)

        if batch  is not None: self.batch  = batch
        if width  is not None: self.width  = width
        if height is not None: self.height = height
        if dtype  is not None: self.dtype  = dtype
