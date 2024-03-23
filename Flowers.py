import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from typing import Union, Type, Literal, Optional

from CycleTimePredictor import CycleTimePredictor
from belashovplot import TiledPlot

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
            self._core = torch.nn.Parameter(torch.rand((core_size, core_size), dtype=torch.complex64))
        else:
            self._core = torch.nn.Parameter(torch.rand((core_size, core_size), dtype=torch.complex64)*torch.exp(2j*torch.pi*torch.rand((core_size, core_size), dtype=torch.float32)))
        # temp = torch.zeros((core_size, core_size), dtype=torch.complex64)
        # temp[int(core_size / 2), int(core_size / 2)] = 1.0
        # self._core = torch.nn.Parameter(temp)

        self._size = size
        self._core_size = size

        self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self._compression_level = compression
        self._convolution_type = convolution

    _size:int
    @property
    def _data_paddings(self):
        right_top = int(self._size / 2)
        left_bottom = int(self._size / 2)
        return left_bottom, right_top, right_top, left_bottom

    _core_size:int
    @property
    def _core_padding(self):
        return int((self._core_size + 1) // 2)
    @property
    def _core_paddings(self):
        left_bottom:int = int((self._size  + 2*self._core_padding - self._core_size) / 2)
        right_top:int = int((self._size + 2*self._core_padding - self._core_size + 1) // 2)
        return left_bottom, right_top, right_top, left_bottom
    @property
    def _core_padding_to_size(self):
        left_bottom: int = int((self._size - self._core_size) / 2)
        right_top: int = int((self._size - self._core_size + 1) // 2)
        return left_bottom, right_top, right_top, left_bottom

    _compression_level:float
    @property
    def _compressed_size(self):
        return int(self._size*self._compression_level)
    @property
    def _compression_padding(self):
        return int((self._size - self._compressed_size + 1) // 2)
    @property
    def _compression_paddings(self):
        left_bottom = int((self._size - self._compressed_size) / 2)
        right_top = int((self._size - self._compressed_size + 1) // 2)
        return left_bottom, right_top, left_bottom, right_top

    @staticmethod
    def _randomize_phase(data:torch.Tensor):
        return data * torch.exp(2j*torch.pi*torch.rand(data.size(), dtype=torch.float32))

    _convolution_type:str
    def compress(self, data:torch.Tensor):
        data = self._randomize_phase(data)
        convolution:Optional[torch.Tensor] = None
        if self._convolution_type == 'fourier':
            padded_data = torch.nn.functional.pad(data, pad=[self._core_padding]*4)
            padded_core = torch.nn.functional.pad(self._core, pad=self._core_paddings)
            padded_data_spectrum = torch.fft.fftshift(torch.fft.fft2(padded_data))
            padded_core_spectrum = torch.fft.fftshift(torch.fft.fft2(padded_core))
            multiplication = padded_core_spectrum * padded_data_spectrum
            convolution = torch.fft.ifftshift(torch.fft.ifft2(multiplication))[:,:,self._core_padding:self._core_padding+self._size,self._core_padding:self._core_padding+self._size]
        elif self._convolution_type == 'convolution':
            padded_data = torch.nn.functional.pad(data, pad=self._data_paddings)
            convolution = torch.nn.functional.conv2d(padded_data, self._core.reshape(1,1,*self._core.size()))
        compressed:torch.Tensor = convolution[:,:,self._compression_padding:self._compression_padding+self._compressed_size,self._compression_padding:self._compression_padding+self._compressed_size]
        return compressed
    def decompress(self, data:torch.Tensor):
        result:Optional[torch.Tensor] = None
        padded_data = torch.nn.functional.pad(data, pad=self._compression_paddings)
        if self._convolution_type == 'fourier':
            padded_core = torch.nn.functional.pad(self._core, pad=self._core_padding_to_size)
            padded_data_spectrum = torch.fft.fftshift(torch.fft.fft2(padded_data))
            padded_core_spectrum = torch.fft.fftshift(torch.fft.fft2(padded_core))
            multiplication = padded_core_spectrum * padded_data_spectrum
            result = torch.fft.ifftshift(torch.fft.ifft2(multiplication))
        elif self._convolution_type == 'convolution':
            padded_data = torch.nn.functional.pad(padded_data, pad=self._data_paddings)
            convolution = torch.nn.functional.conv2d(padded_data, self._core.reshape(1,1,*self._core.size()))
            result = convolution[:,:,self._compression_padding:self._compression_padding+self._compressed_size,self._compression_padding:self._compression_padding+self._compressed_size]
        return result
    def test(self, samples:int=3):
        with torch.no_grad():
            images, labels = next(iter(self._data_test))
            images = images[:samples]
            compressed_list = self.compress(images)
            decompressed_list = self.decompress(compressed_list)

        plot = TiledPlot(14, 14*9/16)
        plot.title("Исходное изображение, сжатое и восстановленное")
        plot.description.top(f'Уровень сжатия: {int(self._compression_level*100)}%, Размер входного изображения {self._size} на {self._size} пикселей.')

        plot.description.column.top('Исходное', 0)
        plot.description.column.top('Сжатое', 1)
        plot.description.column.top('Восстановленное', 2)

        for row, (image, compressed, decompressed) in enumerate(zip(images, compressed_list, decompressed_list)):
            axes = plot.axes.add(0, row)
            axes.imshow(image.abs().squeeze(), aspect='auto', cmap='gray')

            axes = plot.axes.add(1, row)
            axes.imshow(compressed.abs().squeeze(), aspect='auto', cmap='gray')

            axes = plot.axes.add(2, row)
            axes.imshow(decompressed.abs().squeeze(), aspect='auto', cmap='gray')

        plot.show()

    _core:torch.nn.Parameter
    @property
    def _decompression_core(self):
        if self._convolution_type == 'fourier':
            pass
        elif self._convolution_type == 'convolution':
            raise Exception('На данный момент существующий алгоритм пиздец какой сложный https://www.notion.so/4da066e852174b1b94401e818a973bd4?pvs=4#076f6fbba3c6415684149472f0860cb4')

    def forward(self, data:torch.Tensor):

        pass


if __name__ == '__main__':
    Compressor = ConvolutionalCompression('Flowers', size=81, compression=0.9, convolution='convolution')
    Compressor.test()