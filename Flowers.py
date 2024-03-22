import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from typing import Union, Type, Literal

from CycleTimePredictor import CycleTimePredictor

class ConvolutionalCompression(torch.Module):
    _data_train : DataLoader
    _data_test  : DataLoader

    def __init__(self, dataset:str, size:int=101, batch:int=64, core_ratio:float=1.0, core_type:Literal['phase','amplitude','combat']='phase'):
        super().__init__()

        redirections = {
            "Flowers" : (datasets.Flowers102, 'split', ['train', 'test'], {
                'root':'data',
                'download':True,
                'transforms': transforms.Compose([
                    transforms.Grayscale(),
                    transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.ConvertImageDtype(torch.complex64)
                ])
            }),
            "MNIST"   : (datasets.MNIST, 'train', [True, False], {
                'root':'data',
                'download':True,
                'transforms':transforms.Compose([
                    transforms.Grayscale(),
                    transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.ConvertImageDtype(torch.complex64)
                ])
            }),
        }
        dataset_type, kwarg, values, kwargs = redirections[dataset]
        train_dataset = dataset_type(**{kwarg:values[0]}, **kwargs)
        test_dataset  = dataset_type(**{kwarg:values[1]}, **kwargs)

        self._data_train = DataLoader(train_dataset, batch_size=batch, shuffle=True)
        self._data_test  = DataLoader(test_dataset,  batch_size=batch, shuffle=True)

        core_size = int(size*core_ratio)
        if core_type == 'phase':
            self._core = torch.nn.Parameter(torch.exp(2j*torch.pi*torch.rand((core_size, core_size), dtype=torch.float32)))
        elif core_type == 'amplitude':
            self._core = torch.nn.Parameter(torch.rand((core_size, core_size), dtype=torch.complex64))
        else:
            self._core = torch.nn.Parameter(torch.rand((core_size, core_size), dtype=torch.complex64)*torch.exp(2j*torch.pi*torch.rand((core_size, core_size), dtype=torch.float32)))

        self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


    _core:torch.nn.Parameter
    def forward(self):
        return