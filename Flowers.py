import matplotlib.pyplot
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from typing import Union, Type, Literal, Optional, Tuple, Callable

from CycleTimePredictor import CycleTimePredictor
from belashovplot import TiledPlot
from math import sqrt
from itertools import product
from Formaters import Format

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

    @property
    def device(self):
        return self._core.device

    def __init__(self, dataset:str, size:int=101, batch:int=64, core_ratio:float=1.0, core_type:Literal['phase','amplitude','combat']='combat', compression:float=0.7, convolution:Literal['fourier', 'convolution']='fourier'):
        super().__init__()

        redirections = {
            "Flowers" : (datasets.Flowers102, 'split', ['test', 'train'], {
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
        # self._core = torch.nn.Parameter(temp)
        self._decompression_core = torch.nn.Parameter(torch.fft.ifft2(1.0 / torch.fft.fftshift(torch.fft.fft2(self._core.clone()))))

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
        return data * torch.exp(2j*torch.pi*torch.rand(data.size(), dtype=torch.float32, device=data.device))

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
        # data = self._randomize_phase(data)
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
            images = images[:samples].to(self.device)
            compressed_list = self.compress(images)
            decompressed_list = self.decompress(compressed_list)

            images = images.cpu()
            compressed_list = compressed_list.cpu()
            decompressed_list = decompressed_list.cpu()

        plot = TiledPlot(14, 14*9/16)
        plot.title("Исходное изображение, сжатое и восстановленное")
        plot.description.top(f'Уровень сжатия: {int(self._compression_level*100)}%, Размер входного изображения {self._size} на {self._size} пикселей.')

        plot.description.row.right('Исходное', 0)
        plot.description.row.right('Сжатое', 1)
        plot.description.row.right('Восстановленное', 2)

        axes = plot.axes.add(0, 0)
        plot.graph.description('Ядро сжатия')
        axes.imshow(torch.abs(self._core.detach()).cpu(), aspect='auto', cmap='viridis')

        axes = plot.axes.add(0, 2)
        plot.graph.description('Ядро расшифровки')
        axes.imshow(torch.abs(self._decompression_core.detach()).cpu(), aspect='auto', cmap='viridis')

        for col, (image, compressed, decompressed) in enumerate(zip(images, compressed_list, decompressed_list), start=1):
            axes = plot.axes.add(col, 0)
            axes.imshow(image.abs().squeeze(), aspect='auto', cmap='gray')

            axes = plot.axes.add(col, 1)
            axes.imshow(compressed.abs().squeeze(), aspect='auto', cmap='gray')

            axes = plot.axes.add(col, 2)
            axes.imshow(decompressed.abs().squeeze(), aspect='auto', cmap='gray')

        plot.show()

    _core:torch.nn.Parameter
    _decompression_core:torch.nn.Parameter
    # @property
    # def _decompression_core(self):
    #     if self._convolution_type == 'fourier':
    #         return torch.fft.ifft2(1.0 / torch.fft.fftshift(torch.fft.fft2(self._core)))
    #     elif self._convolution_type == 'convolution':
    #         raise Exception('На данный момент существующий алгоритм пиздец какой сложный https://www.notion.so/4da066e852174b1b94401e818a973bd4?pvs=4#076f6fbba3c6415684149472f0860cb4')
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

    @staticmethod
    def _normalize(data:torch.Tensor):
        min_val = data.min(-1)[0].min(-1)[0]
        max_val = data.max(-1)[0].max(-1)[0]
        return (data - min_val[:, :, None, None]) / (max_val[:, :, None, None] - min_val[:, :, None, None])

    @staticmethod
    def border_normalization(a:torch.Tensor):
        return

    @staticmethod
    def mean_square_deviation(a:torch.Tensor, b:torch.Tensor):
        return torch.sqrt(torch.mean((a - b)**2) / (a.numel() - 1))
    @staticmethod
    def mean_square_deviation_relative(a:torch.Tensor, b:torch.Tensor):
        return torch.sqrt(torch.mean(((a - b)/(a + 1.0))**2) / (a.numel() - 1))
    @staticmethod
    def max_deviation(a:torch.Tensor, b:torch.Tensor):
        return ((a-b)**2).max(-1)[0].max(-1)[0]
    @staticmethod
    def max_deviation_relative(a:torch.Tensor, b:torch.Tensor):
        return (((a - b)/(a + 1.0)) ** 2).max(-1)[0].max(-1)[0]
    @staticmethod
    def reverse_peak_signal_to_noise_ration(a:torch.Tensor, b:torch.Tensor):
        return 1.0 / (10 * torch.log10(1.0 / torch.nn.functional.mse_loss(a, b) + 1.0))

    def _epoch(self, comparison_function:Callable, loss_function:Callable, optimizer:torch.optim.Optimizer):
        loss_buffer_size = 10
        loss_buffer_position = 0
        loss_buffer = [0 for i in range(loss_buffer_size)]
        def show_loss():
            return f'Текущий лосс: {Format.Scientific(loss_buffer[loss_buffer_position], "", 2)}'
        def show_average_loss():
            return f'Средний лосс за {loss_buffer_size} операций: {Format.Scientific(sum(loss_buffer)/loss_buffer_size, "", 2)}'

        for batch, (images, labels) in enumerate(CycleTimePredictor(self._data_train, [show_loss, show_average_loss])):
            if not isinstance(images, torch.Tensor):
                raise Exception
            images = images.to(self.device)
            decompressed = self.forward(images)
            images = images.abs()
            decompressed = decompressed.abs()
            deviations = comparison_function(images, decompressed)
            loss = loss_function(deviations)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # prev_loss = loss_buffer[loss_buffer_position]
            loss_buffer_position = loss_buffer_position + 1 if loss_buffer_position < loss_buffer_size - 1 else 0
            loss_buffer[loss_buffer_position] = loss.item()
            # curr_loss = loss_buffer[loss_buffer_position]

            # if curr_loss > prev_loss:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] *= 0.5
            # else:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] *= 1.1


    def _accuracy(self, comparison_function:Callable, loss_function:Callable):
        with torch.no_grad():
            average_loss = 0.0
            counter:int = 0
            def show_average_loss():
                return f'Средний лосс: {Format.Scientific(average_loss / (counter if counter > 0 else 1), "", 2)}'
            for batch, (images, labels) in enumerate(CycleTimePredictor(self._data_test, [show_average_loss])):
                if not isinstance(images, torch.Tensor):
                    raise Exception
                images = images.to(self.device)
                decompressed = self.forward(images)
                # images = self._normalize(images.abs())
                # decompressed = self._normalize(decompressed.abs())
                images = images.abs()
                decompressed = decompressed.abs()
                deviations = comparison_function(images, decompressed)
                loss = loss_function(deviations)

                average_loss += loss.item()
                counter += 1
    def optimize(self, epochs:int=1, loss_function:Union[Callable,Literal['Mean']]='Mean', comparison_function:Union[Callable,Literal['MSD', 'MSDR', 'Max', 'MaxR', 'RPSNR']]='MSD', optimizer:type(torch.optim.Optimizer)=None, **optimizer_kwargs):
        if isinstance(loss_function, str):
            if loss_function == 'Mean':
                loss_function = torch.mean
            else:
                loss_function = torch.mean
        if isinstance(comparison_function, str):
            if comparison_function == 'MSD':
                comparison_function = self.mean_square_deviation
            elif comparison_function == 'NSDR':
                comparison_function = self.mean_square_deviation_relative
            elif comparison_function == 'Max':
                comparison_function = self.max_deviation
            elif comparison_function == 'MaxR':
                comparison_function = self.max_deviation_relative
            elif comparison_function == 'RPSNR':
                comparison_function = self.reverse_peak_signal_to_noise_ration
            else:
                comparison_function = self.mean_square_deviation
        if optimizer is None:
            optimizer = torch.optim.Adam
            optimizer_kwargs = {
                'lr': 0.01,
            }
        optimizer = optimizer(params=self.parameters(), **optimizer_kwargs)

        self._accuracy(comparison_function, loss_function)
        for epoch in range(epochs):
            self.train()
            self._epoch(comparison_function, loss_function, optimizer)
            self.eval()
            self._accuracy(comparison_function, loss_function)


if __name__ == '__main__':
    Compressor = ConvolutionalCompression('Flowers', batch=404, size=511, compression=0.1, core_ratio=1.0, core_type='phase')
    Compressor.test()

    rate = 0.01
    while True:
        print(f'Rate: {Format.Scientific(rate, "", 2)}')
        Compressor.optimize(optimizer=torch.optim.Adam, lr=rate, amsgrad=True, epochs=3, comparison_function='MSD')
        Compressor.test()
        Compressor.optimize(optimizer=torch.optim.Adam, lr=rate, amsgrad=True, epochs=3, comparison_function='MSD')
        Compressor.test()
        Compressor.optimize(optimizer=torch.optim.Adam, lr=rate, amsgrad=True, epochs=3, comparison_function='MSD')
        Compressor.test()
        rate /= 2

