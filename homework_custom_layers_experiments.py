import torch
from utils.training_utils import get_cifar_loaders, count_parameters, train_model
from utils.visualization_utils import compare_models
from models.custom_layers import CustomActivation, CustomConv2d, SoftmaxPooling, SimpleSelfAttention
from torch import nn
from torch.nn import functional as F


def test_customs():
    """3.1 задание"""
    test_custom_conv2d()
    test_custom_activation()
    test_softmax_pooling()
    test_attention()
    torch.manual_seed(42)


def test_custom_conv2d():
    """Сравнение кастомного слоя с нормализацией весов с сверточным слоем"""
    x = torch.randn(1, 3, 8, 8)
    conv_std = torch.nn.Conv2d(3, 6, 3, stride=1, padding=1, bias=True)

    conv_custom = CustomConv2d(3, 6, 3, stride=1, padding=1)
    with torch.no_grad():
        conv_custom.weight.copy_(conv_std.weight)
        conv_custom.bias.copy_(conv_std.bias)

    out_std = conv_std(x)
    out_custom = conv_custom(x)

    print("Conv2d output:", (out_std).abs().max().item())
    print("Custom Conv2d output:", (out_custom).abs().max().item())


def test_custom_activation():
    """Сравнение кастомной функции активаци с ReLu"""
    x = torch.randn(10)
    act_custom = CustomActivation()
    act_std = torch.nn.ReLU()

    out_custom = act_custom(x)
    out_std = act_std(x)

    print("Activation output sample:", out_custom[:5])
    print("ReLU output sample:", out_std[:5])


def test_softmax_pooling():
    """Сравнение кастомного softmax пуллинга с MaxPool"""
    x = torch.randn(1, 3, 8, 8)
    pool_custom = SoftmaxPooling(kernel_size=2, stride=2)
    out_custom = pool_custom(x)
    pool_std = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    out_std = pool_std(x)
    print("SoftmaxPooling output shape:", out_custom.shape)
    print("MaxPooling output shape:", out_std.shape)


def test_attention():
    x = torch.randn(1, 8, 4, 4)
    attention = SimpleSelfAttention(8)
    out = attention(x)
    print("Attention output shape:", out.shape)


class ResidualBlock(nn.Module):
    """Базовый residual блок"""
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = out.view(out.size(0), -1)
        out = F.relu(out)
        return out


class BottleNeckResidualBlock(nn.Module):
    """BottleNeck residual блок"""
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, padding=1, stride=1, bias=False, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*4, bias=False, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels*4)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels*4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*4, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels*4),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = out.view(out.size(0), -1)
        out = F.relu(out)
        return out
    

class WideResidualBlock(nn.Module):
    """Базовый residual блок, но с коэффициентом расширения"""
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, wide_factor=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels * wide_factor, kernel_size, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels * wide_factor)
        self.conv2 = nn.Conv2d(out_channels * wide_factor, out_channels * wide_factor, kernel_size, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * wide_factor)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * wide_factor:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * wide_factor, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * wide_factor),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = out.view(out.size(0), -1)
        out = F.relu(out)
        return out


def compare_all_blocks_cifar():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_cifar_loaders(batch_size=64)
    base_block = ResidualBlock(3, 10).to(device)
    bn_block = BottleNeckResidualBlock(3,10).to(device)
    wide_block = WideResidualBlock(3,10).to(device)

    print(f"Base block parameters: {count_parameters(base_block)}")
    print(f"BottleNeck block parameters: {count_parameters(bn_block)}")
    print(f"wide block parameters: {count_parameters(wide_block)}")

    bace_history = train_model(base_block, train_loader, test_loader, epochs=5, device=str(device))
    bn_history = train_model(bn_block, train_loader, test_loader, epochs=5, device=str(device))
    wide_history = train_model(wide_block, train_loader, test_loader, epochs=5, device=str(device))
    compare_models(bace_history, bn_history, "plots/task_3/base_bn_blocks.png", l1="base_block", l2="bottleneck")
    compare_models(bn_history, wide_history, "plots/task_3/bn_wide_blocks.png", l1="bottleneck", l2="wide block")


if __name__ == "__main__":
    test_customs()
    compare_all_blocks_cifar()