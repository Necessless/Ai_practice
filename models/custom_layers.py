from torch import nn
import torch


class CustomConv2d(nn.Module):
    """Кастомный свёрточный слой с добавочной нормализацией весов по фильтру"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        weight_squared = self.weight ** 2
        weight_sum = weight_squared.sum(dim=(1, 2, 3), keepdim=True)
        weight_norms = torch.sqrt(weight_sum)
        weight_norm = self.weight / (weight_norms + 1e-6)
        return torch.nn.functional.conv2d(x, weight_norm, self.bias, stride=self.stride, padding=self.padding)


class SimpleSelfAttention(nn.Module):
    """Self-attention класс для CNN нейросети"""
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()

        Q = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        K = self.key_conv(x).view(B, -1, H * W)
        V = self.value_conv(x).view(B, -1, H * W)

        attention = torch.nn.functional.softmax(torch.bmm(Q, K) / (K.size(1) ** 0.5), dim=-1)
        out = torch.bmm(V, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        return self.gamma * out + x


class CustomActivation(nn.Module):
    """Немного изменил функцию сигмоид"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x) / (1 + torch.exp(-x))


class SoftmaxPooling(nn.Module):
    """Софтмакс пуллинг"""
    def __init__(self, kernel_size=2, stride=2, temperature=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.temperature = temperature

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        nets = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)  # разбиваем вход на сетку
        nets = nets.contiguous()
        nets = nets.view(batch_size, channels, -1, self.kernel_size * self.kernel_size)  # форматируем ячейки
        weights = torch.nn.functional.softmax(nets / self.temperature, dim=-1)  # вычисляем вероятности через софтмакс
        pooled = torch.sum(weights * nets, dim=-1)  # вычисляем среднее во всей сетке 

        H_out = (height - self.kernel_size) // self.stride + 1
        W_out = (width - self.kernel_size) // self.stride + 1

        pooled = pooled.view(batch_size, channels, H_out, W_out)  # возвращаем полученное значение в новом формате
        return pooled
