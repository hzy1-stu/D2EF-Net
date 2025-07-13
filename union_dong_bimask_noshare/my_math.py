import torch
import numpy as np
import math
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F

# Centered fft2
# fft2c这个函数是傅里叶变换，里面的fftshift是将图像的频率分布转移到中心位置
# 如果要将图像转换成K空间频率数据，就必须采取以下三种变换
def fft2c(img):

    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))


# Centered ifft2
# ifft2这个函数就是逆傅里叶变换
def ifft2c(img):

    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img)))


# Centered fft2 of Tensor
def torch_fft2c(img):

    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img)))


# Centered ifft2 of Tensor
def torch_ifft2c(img):
    """ Centered ifft2 of Tensor"""
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(img)))


# Convert complex to real
# 把实数和复数提取出来，最后按照指定维度拼接到一起（也就是实部和虚部分离）
# 至于为什么拼接到第二个维度上，是因为数据一般是（batch， channel，height， wide），所以要拼接到第二个维度上
def complex2real(x,mask=False):
	x_real = torch.real(x)
	x_imag = torch.imag(x)
	out = torch.cat((x_real,x_imag), dim = -3)

	return out


# Convert real to complex
# 傅立叶空间是复数空间，包括幅度和相位两个信息
def real2complex(x, mask=False):
    idx = x.shape[-3] // 2
    if mask:
        out = x * (1+1j)
    else:
        out = torch.complex(x[:, :idx, :, :], x[:, idx:, :, :])

    return out

# sos（平方和根）就是将多通道变成单通道。医生最后看的就是单通道的图像
# Convert multi-channel image to single-channel image
def sos(im):
	y = 0

	for x in range(im.shape[-3]):
		a = torch.abs(im[:, x, :, :])
		a1 = torch.square(a)
		y = y + a1

	res = torch.sqrt(y)

	return res


# calculate RMSE value
def get_rmse(prediction, target):
    diff = torch.abs(target - prediction)
    diff = diff ** 2
    diff = torch.mean(diff)
    diff = torch.sqrt(diff)
    return diff


# calculate PSNR value
def get_psnr(prediction, target):
    mse = torch.mean((prediction/torch.max(target) - target/torch.max(target)) ** 2)
    if mse == 0:
        return 0.5
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# ssim #
def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


# calculate SSIM value
def get_ssim(prediction, target):
    prediction = torch.reshape(prediction, (1, 1, prediction.shape[0], prediction.shape[1]))
    target = torch.reshape(target, (1, 1, target.shape[0], target.shape[1]))
    window_size = 11
    size_average = True
    channel = 1
    window = create_window(window_size, channel)

    (_, channel, _, _) = prediction.size()

    if prediction.is_cuda:
        window = window.cuda(prediction.get_device())
    window = window.type_as(prediction)

    return _ssim(prediction, target, window, window_size, channel, size_average)