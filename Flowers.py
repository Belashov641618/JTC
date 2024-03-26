import copy
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from typing import Union, Literal, Optional, Tuple, Callable

from CycleTimePredictor import CycleTimePredictor
from belashovplot import TiledPlot
from math import sqrt
from itertools import product
from Formaters import Format
import numpy

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
    def get_images(self, count:int, test:bool=True):
        if test:    data = self._data_test
        else:       data = self._data_train
        data = iter(data)

        images, labels = next(data)
        if images.size(0) > count:
            images_ = images[:count, :,:,:]
            count = 0
        else:
            images_ = images
            count -= images.size(0)

        while count > 0:
            images, labels = next(data)
            if images.size(0) > count:
                images_ = torch.cat((images_, images[:count, :, :, :]), dim=0)
                count = 0
            else:
                images_ = torch.cat((images_, images), dim=0)
                count -= images.size(0)

        images_ = images_.to(self.device)
        return images_

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
            self._core = torch.exp(2j*torch.pi*torch.rand((core_size, core_size), dtype=torch.float32))
        elif core_type == 'amplitude':
            self._core = torch.rand((core_size, core_size), dtype=torch.float32).to(torch.complex64)
        else:
            self._core = torch.rand((core_size, core_size), dtype=torch.complex64)*torch.exp(2j*torch.pi*torch.rand((core_size, core_size), dtype=torch.float32))
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
        self._accuracy = None

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

        a_paddings = [0 for _ in range(4)]
        b_paddings = [0 for _ in range(4)]

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

    _compression_core:torch.nn.Parameter
    @property
    def _core(self):
        return (1.0 / (1.0 + torch.exp(-torch.abs(self._compression_core)))) * torch.exp(1j * torch.angle(self._compression_core))
    @_core.setter
    def _core(self, data:torch.Tensor):
        self._compression_core = torch.nn.Parameter((-torch.log(1.0 / data - 1.0)) * torch.exp(1j * torch.angle(data)))
    _decompression_core:torch.nn.Parameter
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

    @property
    def params(self):
        class Selector:
            _self : ConvolutionalCompression
            def __init__(self, _self:ConvolutionalCompression):
                self._self = _self
            @property
            def compression(self):
                return self._self._compression_core
            @property
            def decompression(self):
                return self._self._decompression_core
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
    @staticmethod
    def reversed_square_cross_correlation(a:torch.Tensor, b:torch.Tensor):
        a = a - torch.mean(a, dim=(2,3), keepdim=True)
        b = b - torch.mean(b, dim=(2,3), keepdim=True)
        return (1.0 - torch.sum(a*b,dim=(2,3)) / torch.sqrt(torch.sum(a*a, dim=(2,3))*torch.sum(b*b, dim=(2,3))))**2

    _accuracy:Optional[float]
    def _iter(self, images:torch.Tensor, comparison_function:Callable, loss_function:Callable):
        images = images.to(self.device)
        decompressed = self.forward(images)
        images = images.abs()
        decompressed = decompressed.abs()
        deviations = comparison_function(images, decompressed)
        loss = loss_function(deviations)
        return loss
    def _epoch(self, comparison_function:Callable, loss_function:Callable, optimizer:torch.optim.Optimizer):
        self._accuracy = None

        loss_history = []

        loss_buffer_size = 10
        loss_buffer_position = 0
        loss_buffer = [0 for _ in range(loss_buffer_size)]
        def show_loss():
            return f'Текущий лосс: {Format.Scientific(loss_buffer[loss_buffer_position], "", 2)}'
        def show_average_loss():
            return f'Средний лосс за {loss_buffer_size} операций: {Format.Scientific(sum(loss_buffer)/loss_buffer_size, "", 2)}'

        self.train()
        for batch, (images, labels) in enumerate(CycleTimePredictor(self._data_train, [show_loss, show_average_loss])):
            if not isinstance(images, torch.Tensor):
                raise TypeError
            loss = self._iter(images, comparison_function, loss_function)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_buffer_position = loss_buffer_position + 1 if loss_buffer_position < loss_buffer_size - 1 else 0
            loss_buffer[loss_buffer_position] = loss.item()

            loss_history.append(loss_buffer[loss_buffer_position])
        self.eval()

        return loss_history
    def accuracy(self, comparison_function:Callable, loss_function:Callable):
        if self._accuracy is None:
            with torch.no_grad():
                average_loss = 0.0
                counter:int = 0
                def show_average_loss():
                    return f'Средний лосс: {Format.Scientific(average_loss / (counter if counter > 0 else 1), "", 2)}'
                for batch, (images, labels) in enumerate(CycleTimePredictor(self._data_test, [show_average_loss])):
                    if not isinstance(images, torch.Tensor):
                        raise TypeError
                    loss = self._iter(images, comparison_function, loss_function)

                    average_loss += loss.item()
                    counter += 1
                self._accuracy = average_loss / counter
        return self._accuracy
    def optimize(self, optimizer:torch.optim.Optimizer, epochs:int=1, loss_function:Union[Callable,Literal['Mean']]='Mean', comparison_function:Union[Callable,Literal['MSD', 'MSDR', 'Max', 'MaxR', 'RPSNR', 'RSCC']]='MSD'):
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
            elif comparison_function == 'RSCC':
                comparison_function = self.reversed_square_cross_correlation
            else:
                comparison_function = self.mean_square_deviation

        optimization_info = OptimizationHistory()

        self.accuracy(comparison_function, loss_function)
        for epoch in range(epochs):
            loss_history = self._epoch(comparison_function, loss_function, optimizer)
            accuracy = self.accuracy(comparison_function, loss_function)
            optimization_info.append(loss_history, accuracy, self)

        return optimization_info

    def get_accuracy(self):
        return self._accuracy

class OptimizationHistory:
    _epochs_loss_histories:list[numpy.ndarray]
    _epochs_accuracy:list[float]
    _epochs_networks:list[ConvolutionalCompression]

    def __init__(self):
        self._epochs_loss_histories = []
        self._epochs_accuracy = []
        self._epochs_networks = []

    def append_loss_history(self, loss_history:Union[numpy.ndarray, list[float]], epoch:int=None):
        if isinstance(loss_history, list):
            loss_history = numpy.array(loss_history)
        if epoch is None or len(self._epochs_loss_histories) <= epoch:
            self._epochs_loss_histories.append(loss_history)
        else:
            self._epochs_loss_histories[epoch] = loss_history
    def append_accuracy(self, accuracy:float, epoch:int=None):
        if epoch is None or len(self._epochs_accuracy) <= epoch:
            self._epochs_accuracy.append(accuracy)
        else:
            self._epochs_accuracy[epoch] = accuracy
    def append_network(self, network:ConvolutionalCompression, epoch:int=None):
        if epoch is None or len(self._epochs_accuracy) <= epoch:
            self._epochs_networks.append(copy.deepcopy(network))
        else:
            self._epochs_networks[epoch] = copy.deepcopy(network)
    def append(self, loss_history:Union[numpy.ndarray, list[float]], accuracy:float, network:ConvolutionalCompression):
        self.append_network(network)
        self.append_accuracy(accuracy)
        self.append_loss_history(loss_history)
    @property
    def length(self):
        return len(self._epochs_networks)

    def __sub__(self, other):
        if isinstance(other, OptimizationHistory):
            temp = OptimizationHistory()
            temp._epochs_loss_histories = self._epochs_loss_histories + other._epochs_loss_histories
            temp._epochs_accuracy       = self._epochs_accuracy       + other._epochs_accuracy
            temp._epochs_networks       = self._epochs_networks       + other._epochs_networks
            return temp
        else: raise TypeError

    def plot(self, images:torch.Tensor, additional_information:str=None):
        with torch.no_grad():
            epochs_reconstructs:list[torch.Tensor] = []
            for network in self._epochs_networks:
                epochs_reconstructs.append(network.forward(images))

        plot = TiledPlot(11.7*1.5, 8.3*1.5)
        plot.title('Подробная история обучения нейронной сети')
        if additional_information is not None:
            plot.description.bottom(additional_information)
        plot.FontLibrary.MultiplyFontSize(0.7)

        image_params = {'aspect':'auto', 'cmap':'gray'}
        def image_function(image_:torch.Tensor):
            return torch.abs(image_).squeeze().cpu()

        def add_image(image_:torch.Tensor, col_:int, row_:int):
            axes_ = plot.axes.add(col_, row_)
            axes_.imshow(image_function(image_), **image_params)

        colors = ['maroon', 'darkorange', 'darkgreen', 'darkslategrey', 'darkblue', 'darkslateblue', 'darkviolet', 'darkmagenta']*(self.length//8 + 1)
        axes = plot.axes.add((1, 0), (self.length, 0))
        axes.grid(True)
        prev_iters = 0
        for color, loss_history in zip(colors, self._epochs_loss_histories):
            iters = numpy.arange(prev_iters+1, prev_iters+loss_history.shape[0]+1)
            prev_iters = iters[-1]
            axes.plot(iters, loss_history, color=color, linewidth=1.0, linestyle='--')
        axes.set_xlim(1, prev_iters)
        plot.title('График лосс-функции')
        plot.graph.label.x('Номер итерации')
        plot.graph.label.y('Лосс')

        axes = plot.axes.add(0, 0)
        axes.grid(True)
        axes.plot(range(1, len(self._epochs_accuracy)+1), self._epochs_accuracy, color='purple', linewidth=1.0, linestyle='--', marker='.')
        plot.title('График точности')
        plot.graph.label.x('Номер эпохи')
        plot.graph.label.y('Точность')

        for row, image in enumerate(images, start=1):
            if isinstance(image, torch.Tensor):
                add_image(image, 0, row)
            plot.description.row.left(f"Пример №{row}", row)
        row_cc = 1 + len(images)
        row_dc = row_cc + 1
        plot.description.row.right('Ядро шифрования', row_cc)
        plot.description.row.right('Ядро дешифрования', row_dc)
        plot.description.row.right('Дешифрованные сжатые изображения', 1, len(images))

        for col, (reconstructions, network) in enumerate(zip(epochs_reconstructs, self._epochs_networks), start=1):
            plot.description.column.top(f"Эпоха №{col}", col)
            for row, decompressed in enumerate(reconstructions, start=1):
                add_image(decompressed, col, row)
            axes = plot.axes.add(col, row_cc)
            axes.imshow(torch.abs(network.core.compression), aspect='auto', cmap='viridis')
            axes = plot.axes.add(col, row_dc)
            axes.imshow(torch.abs(network.core.decompression), aspect='auto', cmap='viridis')

        plot.show()

def rate_test():
    compression_core_rates = numpy.logspace(0, -9, 10, base=10)
    decompression_core_rates = numpy.logspace(0, -9, 10, base=10)

    accuracies = []

    for (n, compression_core_rate), (m, decompression_core_rate) in product(enumerate(compression_core_rates),
                                                                            enumerate(decompression_core_rates)):
        compressor = ConvolutionalCompression('Flowers', batch=404, size=511, compression=0.1)
        compression_group = {'params': compressor.params.compression, 'lr': compression_core_rate}
        decompression_group = {'params': compressor.params.decompression, 'lr': decompression_core_rate}
        optimizer = torch.optim.Adam([compression_group, decompression_group])

        info = compressor.optimize(optimizer, epochs=10, comparison_function='RSCC')
        info.plot(compressor.get_images(3), f"Скорость обучения ядра шифровки: {Format.Scientific(compression_core_rate, '', 3)}\nСкорость обучения ядра дешифровки: {Format.Scientific(decompression_core_rate, '', 3)}")

        accuracies.append(compressor.get_accuracy())

    print(accuracies)

def train():
    compressor = ConvolutionalCompression('Flowers', batch=64, size=511, compression=0.9, core_ratio=1.0)
    compressor.test()

if __name__ == '__main__':
    # rate_test()
    train()

