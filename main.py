import matplotlib.patches as patches
from belashovplot import TiledPlot
from PIL import Image
import torch
import numpy
from torchvision.transforms.functional import resize

def wavelet(N:int, theta:float, phase:float, frequency:float, amplitude:float, sigma_x:float=1.0, sigma_y:float=1.0, norm:bool=True):
    x_mesh, y_mesh = torch.meshgrid(torch.linspace(-1,+1,N), torch.linspace(-1,+1,N), indexing='xy')
    x_mesh, y_mesh = x_mesh*numpy.cos(theta) - y_mesh*numpy.sin(theta), x_mesh*numpy.sin(theta) + y_mesh*numpy.cos(theta)

    wave = amplitude*torch.sin(frequency*x_mesh + phase)
    gauss = (1.0 / (2.0*torch.pi*sigma_x*sigma_y)) * torch.exp(-(x_mesh**2)/(2*sigma_x**2)) * torch.exp(-(y_mesh**2)/(2*sigma_y**2))


    result = (wave + amplitude) * gauss
    if norm:
        result = (result - result.min()) / (result.max() - result.min())

    return result


if __name__ == '__main__':
    picture = Image.open('letters.jpg')
    picture = torch.mean(torch.reshape(torch.tensor(picture.getdata(), dtype=torch.float32, requires_grad=False), (picture.height, picture.width, 3)), dim=2)
    size = min(*picture.size())
    w0 = int((picture.size()[0] - size)/2)
    h0 = int((picture.size()[1] - size)/2)
    picture = picture[w0:w0+size, h0:h0+size]
    picture = (picture - picture.min()) / (picture.max() - picture.min())

    N = 1023
    f = resize(picture.reshape(1,1,size,size), [N, N], antialias=True).squeeze().to(torch.complex64)
    h = torch.exp(2j*torch.pi*torch.rand((N,N), dtype=torch.float32, requires_grad=False))
    c = f * h
    # c = f
    # r = torch.rand((N,N), dtype=torch.float32, requires_grad=False) * torch.exp(2j*torch.pi*torch.rand((N,N), dtype=torch.float32, requires_grad=False))
    # r = torch.exp(2j*torch.pi*torch.rand((N,N), dtype=torch.float32, requires_grad=False))
    # r = torch.rand((N,N), dtype=torch.float32, requires_grad=False)**2 * torch.exp(2j*torch.pi*torch.rand((N,N), dtype=torch.float32, requires_grad=False))
    # r = wavelet(N, -numpy.pi/6, numpy.pi/4, 60.0, 1.0, 0.7, 0.5).to(torch.complex64)
    # r = torch.zeros((N,N),dtype=torch.complex64, requires_grad=False)
    # r[int(N/2), int(N/2)] = 1.0
    x_mesh, y_mesh = torch.meshgrid(torch.linspace(-1,+1,N), torch.linspace(-1,+1,N), indexing='xy')
    radius = torch.sqrt(x_mesh**2 + y_mesh**2)
    r = (radius / radius.max())**0.01 * torch.exp(2j*torch.pi*torch.rand((N,N), dtype=torch.float32, requires_grad=False))

    space_ration = 0.0
    padding_ratio = 0.3
    crop_scale = 0.3

    space = int(N + space_ration * N)
    padding = int(N*padding_ratio)
    micro_shift = int(N*(1.0 - crop_scale)/2)
    shift = padding + micro_shift
    size = int(N*crop_scale)

    total_y = 2*padding + 3*N + 2*space
    total_x = 2*padding + N

    shift_c = padding + N + space
    shift_r = shift_c + N + space

    tiled = torch.zeros((total_x, total_y), dtype=torch.complex64, requires_grad=False)
    tiled[padding:padding+N, shift_c:shift_c+N] = c
    tiled[padding:padding+N, shift_r:shift_r+N] = r

    spectrum = torch.fft.fftshift(torch.fft.fft2(tiled))
    intensity = torch.abs(spectrum) ** 2
    result = torch.fft.fftshift(torch.fft.fft2(intensity))
    cropped = result[shift:shift+size, shift_r+micro_shift:shift_r+micro_shift+size]

    a = shift_r + micro_shift
    b = a + size
    c = padding + micro_shift
    d = c + size
    intensity_ = torch.zeros((total_x, total_y), dtype=torch.complex64, requires_grad=False)
    intensity_[c:d,a:b] = result[c:d,a:b]
    intensity_ = torch.fft.ifft2(torch.fft.ifftshift(intensity_))

    tiled_ = torch.zeros((total_x, total_y), dtype=torch.complex64, requires_grad=False)
    tiled_[padding:padding+N, shift_r:shift_r+N] = r
    tiled_spectrum = torch.fft.fftshift(torch.fft.fft2(tiled_))*intensity_
    decrypted = (torch.fft.fft2(tiled_spectrum))

    cropped_ = torch.abs(decrypted[padding:padding+N, shift_c:shift_c+N])

    plot = TiledPlot(13, 13*9/16)
    plot.FontLibrary.MultiplyFontSize(0.7)
    plot.title('Фурье образ')
    plot.width_to_height(total_y/total_x)

    axes = plot.axes.add((0, 0))
    axes.imshow(torch.abs(tiled), cmap='gray', aspect='auto')
    plot.graph.title('Совмещённое изображение')
    rectangle = patches.Rectangle((shift_c, padding), N, N, linewidth=1.0, edgecolor='maroon', facecolor='none', linestyle='--')
    axes.add_patch(rectangle)

    axes = plot.axes.add((0, 1))
    axes.imshow(intensity, cmap='gray', aspect='auto')
    plot.graph.title('Мощьностной спектр')
    rectangle_ = patches.Rectangle((padding, padding), 3*N + 2*space, N, linewidth=1.0, edgecolor='maroon', facecolor='none', linestyle='--')
    axes.add_patch(rectangle_)

    axes = plot.axes.add((0, 2))
    axes.imshow(torch.abs(result), cmap='gray', aspect='auto', vmax=torch.abs(cropped).max(), vmin=torch.abs(cropped).min())
    plot.graph.title('Модуль результата')
    rectangle__ = patches.Rectangle((shift_r+micro_shift, padding+micro_shift), size, size, linewidth=1.0, edgecolor='green', facecolor='none', linestyle='--')
    axes.add_patch(rectangle__)
    rectangle = patches.Rectangle((shift_r, padding), N, N, linewidth=1.0, edgecolor='maroon', facecolor='none', linestyle='--')
    axes.add_patch(rectangle)


    axes = plot.axes.add((1, 0))
    axes.imshow(torch.abs(tiled_), cmap='gray', aspect='auto')
    plot.graph.title('Ключ шифрования')

    axes = plot.axes.add((1, 1))
    axes.imshow(torch.abs(tiled_spectrum), cmap='gray', aspect='auto')
    plot.graph.title('Спектр расшифровки')
    rectangle___ = patches.Rectangle((padding, padding), 3*N + 2*space, N, linewidth=1.0, edgecolor='maroon', facecolor='none', linestyle='--')
    axes.add_patch(rectangle___)

    axes = plot.axes.add((1, 2))
    axes.imshow(torch.abs(decrypted), cmap='gray', aspect='auto', vmax=cropped_.max(), vmin=cropped_.min())
    plot.graph.title('Модуль результата')

    # axes = plot.axes.add((2, 1))
    # axes.imshow(torch.abs(intensity_), cmap='gray', aspect='auto')
    # plot.graph.title('Обрезанный мощностной спектр')

    plot.show()