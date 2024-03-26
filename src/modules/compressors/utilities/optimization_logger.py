import numpy
import torch
import copy

from typing import Union
from belashovplot import TiledPlot

from ..AbstractCompressor import AbstractCompressor

class OptimizationLogger:
    _epochs_loss_histories:list[numpy.ndarray]
    _epochs_accuracy:list[float]
    _epochs_networks:list[AbstractCompressor]

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
    def append_network(self, network:AbstractCompressor, epoch:int=None):
        if epoch is None or len(self._epochs_accuracy) <= epoch:
            self._epochs_networks.append(copy.deepcopy(network))
        else:
            self._epochs_networks[epoch] = copy.deepcopy(network)
    def append(self, loss_history:Union[numpy.ndarray, list[float]], accuracy:float, network:AbstractCompressor):
        self.append_network(network)
        self.append_accuracy(accuracy)
        self.append_loss_history(loss_history)
    @property
    def length(self):
        return len(self._epochs_networks)

    def __sub__(self, other):
        if isinstance(other, OptimizationLogger):
            temp = OptimizationLogger()
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