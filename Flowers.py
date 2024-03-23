import matplotlib.pyplot
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from typing import Union, Type, Literal, Optional, Tuple, Callable

from CycleTimePredictor import CycleTimePredictor
from belashovplot import TiledPlot
from math import sqrt
from itertools import product

def plot_batch(batch:torch.Tensor, name:str, ratio:float=16/9):
    amount = batch.size(0)
    rows = int(sqrt(amount/ratio))
    cols = amount // rows
    if cols * rows < amount: cols += 1

    plot = TiledPlot(14, 14/ratio)
    plot.title(name)

    for image, (col, row) in zip(batch, product(range(cols), range(rows))):
        image = torch.abs(image.squeeze())
        axes = plot.axes.add(row, col)
        axes.imshow(image, aspect='auto', cmap='gray')

    plot.show()

class ConvolutionalCompression(torch.nn.Module):
    _data_train : DataLoader
    _data_test  : DataLoader

    def __init__(self, dataset:str, size:int=101, batch:int=64, core_ratio:float=1.0, core_type:Literal['phase','amplitude','combat']='combat', compression:float=0.7, convolution:Literal['fourier', 'convolution']='convolution'):
        super().__init__()

        redirections = {
            "Flowers" : (datasets.Flowers102, 'split', ['train', 'test'], {
                'root':'data',
                'download':True,
                'transform': transforms.Compose([
                    transforms.Grayscale(),
                    transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.ToTensor(),
                    transforms.ConvertImageDtype(torch.complex64)
                ])
            }),
            "MNIST"   : (datasets.MNIST, 'train', [True, False], {
                'root':'data',
                'download':True,
                'transform':transforms.Compose([
                    transforms.Grayscale(),
                    transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.ToTensor(),
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
            self._core = torch.nn.Parameter(torch.rand((core_size, core_size), dtype=torch.float32).to(torch.complex64))
        else:
            self._core = torch.nn.Parameter(torch.rand((core_size, core_size), dtype=torch.complex64)*torch.exp(2j*torch.pi*torch.rand((core_size, core_size), dtype=torch.float32)))
        # temp = torch.zeros((core_size, core_size), dtype=torch.complex64)
        # temp[int(core_size / 2), int(core_size / 2)] = 1.0
        # temp[0, 0] = 0.5
        # self._core = torch.nn.Parameter(temp)

        self._size = size
        self._core_size = size

        self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self._compression_level = compression
        self._convolution_type = convolution
        self._fourier_padding = True

    _size:int
    _core_size:int
    _compression_level:float
    @property
    def _compressed_size(self):
        return int(self._size*sqrt(self._compression_level))

    _fourier_padding:bool

    @property
    def _compressed_frame(self):
        padding = (self._size - self._compressed_size) // 2
        return padding, padding+self._compressed_size, padding, padding+self._compressed_size

    @staticmethod
    def fix_shape(a:torch.Tensor):
        while len(a.size()) != 4: a = a.unsqueeze(0)
        return a
    @staticmethod
    def pad_to_same_size(a:torch.Tensor, b:torch.Tensor):
        a = ConvolutionalCompression.fix_shape(a)
        b = ConvolutionalCompression.fix_shape(b)

        a_paddings = [0 for i in range(4)]
        b_paddings = [0 for i in range(4)]

        for i in range(2):
            n = i + 2
            if a.size(2) >= b.size(2):
                b_paddings[2*i + 0] = (a.size(n) - b.size(n)) // 2
                b_paddings[2*i + 1] = a.size(n) - b.size(n) - b_paddings[0]
            else:
                a_paddings[2*i + 0] = (b.size(n) - a.size(n)) // 2
                a_paddings[2*i + 1] = b.size(n) - a.size(n) - a_paddings[0]

        a_frame = (a_paddings[0], a_paddings[0] + a.size(2), a_paddings[2], a_paddings[2] + a.size(3))
        b_frame = (b_paddings[0], b_paddings[0] + b.size(2), b_paddings[2], b_paddings[2] + b.size(3))

        if sum(a_paddings) > 0: a = torch.nn.functional.pad(a, pad=a_paddings)
        if sum(b_paddings) > 0: b = torch.nn.functional.pad(b, pad=b_paddings)

        return a, b, a_frame, b_frame
    @staticmethod
    def un_pad_from_frame(a:torch.Tensor, a_frame:Tuple[int,int,int,int]):
        a = ConvolutionalCompression.fix_shape(a)
        return a[:,:,a_frame[0]:a_frame[1],a_frame[2]:a_frame[3]]
    @staticmethod
    def pad_to_size(a:torch.Tensor, size:int):
        a = ConvolutionalCompression.fix_shape(a)
        left = (size - a.size(2)) // 2
        right = size - a.size(2) - left
        top = (size - a.size(3)) // 2
        bottom = size - a.size(3) - top
        if left + top > 0: a = torch.nn.functional.pad(a, pad=(left,right, top,bottom))
        a_frame = (left, left+a.size(2), top, top+a.size(3))
        return a, a_frame

    @staticmethod
    def _randomize_phase(data:torch.Tensor):
        return data * torch.exp(2j*torch.pi*torch.rand(data.size(), dtype=torch.float32))

    _convolution_type:str
    def _convolution(self, data:torch.Tensor, core:torch.Tensor):
        if self._convolution_type == 'fourier':
            core, data, core_frame, data_frame = self.pad_to_same_size(core, data)
            core_spectrum = torch.fft.fftshift(torch.fft.fft2(core))
            data_spectrum = torch.fft.fftshift(torch.fft.fft2(data))
            convolution = torch.fft.ifftshift(torch.fft.ifft2(data_spectrum*core_spectrum))
            convolution = self.un_pad_from_frame(convolution, data_frame)
            return convolution
        elif self._convolution_type == 'convolution':
            return None

    def compress(self, data:torch.Tensor):
        data = self._randomize_phase(data)
        convolution = self._convolution(data, self._core)
        compressed = self.un_pad_from_frame(convolution, self._compressed_frame)
        return compressed
    def decompress(self, compressed:torch.Tensor):
        compressed, compressed_frame = self.pad_to_size(compressed, self._size)
        decompressed = self._convolution(compressed, self._decompression_core)
        return decompressed
    def test(self, samples:int=3):
        with torch.no_grad():
            images, labels = next(iter(self._data_test))
            images = images[:samples]
            compressed_list = self.compress(images)
            decompressed_list = self.decompress(compressed_list)

        plot = TiledPlot(14, 14*9/16)
        plot.title("Исходное изображение, сжатое и восстановленное")
        plot.description.top(f'Уровень сжатия: {int(self._compression_level*100)}%, Размер входного изображения {self._size} на {self._size} пикселей.')

        plot.description.row.right('Исходное', 0)
        plot.description.row.right('Сжатое', 1)
        plot.description.row.right('Восстановленное', 2)

        axes = plot.axes.add(0, 0)
        plot.graph.description('Ядро сжатия')
        axes.imshow(torch.abs(self._core.detach()), aspect='auto', cmap='viridis')

        axes = plot.axes.add(0, 2)
        plot.graph.description('Ядро расшифровки')
        axes.imshow(torch.abs(self._decompression_core.detach()), aspect='auto', cmap='viridis')

        for col, (image, compressed, decompressed) in enumerate(zip(images, compressed_list, decompressed_list), start=1):
            axes = plot.axes.add(col, 0)
            axes.imshow(image.abs().squeeze(), aspect='auto', cmap='gray')

            axes = plot.axes.add(col, 1)
            axes.imshow(compressed.abs().squeeze(), aspect='auto', cmap='gray')

            axes = plot.axes.add(col, 2)
            axes.imshow(decompressed.abs().squeeze(), aspect='auto', cmap='gray')

        plot.show()

    _core:torch.nn.Parameter
    @property
    def _decompression_core(self):
        if self._convolution_type == 'fourier':
            return torch.fft.ifft2(1.0 / torch.fft.fftshift(torch.fft.fft2(self._core)))
        elif self._convolution_type == 'convolution':
            raise Exception('На данный момент существующий алгоритм пиздец какой сложный https://www.notion.so/4da066e852174b1b94401e818a973bd4?pvs=4#076f6fbba3c6415684149472f0860cb4')
    @property
    def core(self):
        class Selector:
            _self : ConvolutionalCompression
            def __init__(self, _self:ConvolutionalCompression):
                self._self = _self
            @property
            def compression(self):
                return self._self._core.clone().detach().cpu()
            @property
            def decompression(self):
                return self._self._decompression_core.clone().detach().cpu()
        return Selector(self)

    def forward(self, data:torch.Tensor):
        return self.decompress(self.compress(data))

    def _epoch(self, loss_function:Callable):
        for batch, (images, labels) in enumerate(CycleTimePredictor(self._data_train)):
            if not isinstance(images, torch.Tensor):
                raise Exception
            decompressed = self.forward(images)
            loss =


        pass
    def _accuracy(self):
        pass
    def train(self, epochs:int=1, loss_function:Union[Callable,Literal['MSE']]= 'MSE'):
        if isinstance(loss_function, str):
            if loss_function == 'MSE':
                loss_function = torch.nn.MSELoss()
            else:
                loss_function = torch.nn.MSELoss()
        for epoch in range(epochs):
            self.train()
            self._epoch(loss_function)
            self.eval()
            self._accuracy()


if __name__ == '__main__':

    Compressor = ConvolutionalCompression('Flowers', size=255, compression=0.95, convolution='fourier', core_ratio=1.0, core_type='combat')
    Compressor.test()

