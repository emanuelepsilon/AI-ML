from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import ExperimentConfig


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels) ** 0.5
        shape = (in_channels, out_channels, modes, modes)
        self.weight_top = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_bottom = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def multiply(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = inputs.shape
        spectrum = torch.fft.rfft2(inputs, norm="ortho")
        output = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            device=inputs.device,
            dtype=torch.cfloat,
        )
        modes_y = min(self.modes, height // 2)
        modes_x = min(self.modes, width // 2 + 1)
        output[:, :, :modes_y, :modes_x] = self.multiply(
            spectrum[:, :, :modes_y, :modes_x],
            self.weight_top[:, :, :modes_y, :modes_x],
        )
        output[:, :, -modes_y:, :modes_x] = self.multiply(
            spectrum[:, :, -modes_y:, :modes_x],
            self.weight_bottom[:, :, :modes_y, :modes_x],
        )
        return torch.fft.irfft2(output, s=(height, width), norm="ortho")


class FourierBlock(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes)
        self.local = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(4, width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mixed = self.spectral(inputs) + self.local(inputs)
        return F.gelu(self.norm(mixed) + inputs)


class FourierNeuralOperator2d(nn.Module):
    def __init__(
        self,
        input_channels: int = 8,
        output_channels: int = 3,
        width: int = 20,
        modes: int = 8,
        layers: int = 4,
        padding: int = 4,
    ) -> None:
        super().__init__()
        self.padding = padding
        self.lift = nn.Conv2d(input_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(FourierBlock(width, modes) for _ in range(layers))
        self.project = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, output_channels, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.lift(inputs)
        features = F.pad(features, (0, self.padding, 0, self.padding))
        for block in self.blocks:
            features = block(features)
        features = features[..., : -self.padding, : -self.padding]
        return inputs[:, :3] + self.project(features)


class ResidualConvBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(4, width),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.gelu(inputs + self.layers(inputs))


class ConvolutionalBaseline(nn.Module):
    def __init__(self, input_channels: int = 8, output_channels: int = 3, width: int = 24) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, width, 5, padding=2),
            nn.GELU(),
            ResidualConvBlock(width, 1),
            ResidualConvBlock(width, 2),
            ResidualConvBlock(width, 4),
            ResidualConvBlock(width, 2),
            nn.Conv2d(width, output_channels, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, :3] + self.network(inputs)


def build_model(name: str, config: ExperimentConfig) -> nn.Module:
    if name == "fno":
        return FourierNeuralOperator2d(
            width=config.fno_width,
            modes=config.fno_modes,
            layers=config.fno_layers,
        )
    if name == "cnn":
        return ConvolutionalBaseline()
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
