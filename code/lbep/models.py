from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn


class ConvNormActivation(nn.Module):
    def __init__(self, dimensions: int, input_channels: int, output_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        convolution = nn.Conv3d if dimensions == 3 else nn.Conv2d
        normalization = nn.InstanceNorm3d if dimensions == 3 else nn.InstanceNorm2d
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(convolution(input_channels, output_channels, kernel_size, padding=kernel_size // 2, bias=False)),
            normalization(output_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.layers(inputs))


class ResidualStage(nn.Module):
    def __init__(self, dimensions: int, channels: int, blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ConvNormActivation(dimensions, channels, channels) for _ in range(blocks)])

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs
        for block in self.blocks:
            outputs = outputs + block(outputs)
        return outputs


class EquitableUNet(nn.Module):
    def __init__(self, input_channels: int = 1, classes: int = 3, widths: Sequence[int] = (32, 64, 128, 256)) -> None:
        super().__init__()
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        previous = input_channels
        for width in widths:
            self.encoders.append(nn.Sequential(ConvNormActivation(3, previous, width), ResidualStage(3, width, 2)))
            self.downsamples.append(nn.Conv3d(width, width, 2, stride=2))
            previous = width
        self.bottleneck = nn.Sequential(ConvNormActivation(3, widths[-1], widths[-1] * 2), ResidualStage(3, widths[-1] * 2, 2))
        decoder_channels = widths[-1] * 2
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for width in reversed(widths):
            self.upsamples.append(nn.ConvTranspose3d(decoder_channels, width, 2, stride=2))
            self.decoders.append(nn.Sequential(ConvNormActivation(3, width * 2, width), ResidualStage(3, width, 2)))
            decoder_channels = width
        self.head = nn.Conv3d(widths[0], classes, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        skips: list[Tensor] = []
        outputs = inputs
        for encoder, downsample in zip(self.encoders, self.downsamples, strict=True):
            outputs = encoder(outputs)
            skips.append(outputs)
            outputs = downsample(outputs)
        outputs = self.bottleneck(outputs)
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips), strict=True):
            outputs = upsample(outputs)
            if outputs.shape[2:] != skip.shape[2:]:
                outputs = torch.nn.functional.interpolate(outputs, size=skip.shape[2:], mode="trilinear", align_corners=False)
            outputs = decoder(torch.cat((outputs, skip), dim=1))
        return cast(Tensor, self.head(outputs))


class DenseLayer(nn.Module):
    def __init__(self, input_channels: int, growth_rate: int) -> None:
        super().__init__()
        self.transform = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, growth_rate * 4, 1, bias=False),
            nn.BatchNorm2d(growth_rate * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate * 4, growth_rate, 3, padding=1, bias=False),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.cat((inputs, self.transform(inputs)), dim=1)


class DenseBlock(nn.Module):
    def __init__(self, layers: int, input_channels: int, growth_rate: int) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        channels = input_channels
        for _ in range(layers):
            modules.append(DenseLayer(channels, growth_rate))
            channels += growth_rate
        self.layers = nn.Sequential(*modules)
        self.output_channels = channels

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.layers(inputs))


class RadiographDenseNet(nn.Module):
    def __init__(self, growth_rate: int = 32, blocks: Sequence[int] = (6, 12, 32, 32)) -> None:
        super().__init__()
        channels = 64
        features: list[nn.Module] = [nn.Conv2d(1, channels, 7, stride=2, padding=3, bias=False), nn.MaxPool2d(3, stride=2, padding=1)]
        for index, count in enumerate(blocks):
            block = DenseBlock(count, channels, growth_rate)
            features.append(block)
            channels = block.output_channels
            if index != len(blocks) - 1:
                next_channels = channels // 2
                features.extend([nn.BatchNorm2d(channels), nn.ReLU(inplace=True), nn.Conv2d(channels, next_channels, 1, bias=False), nn.AvgPool2d(2)])
                channels = next_channels
        self.features = nn.Sequential(*features)
        self.normalization = nn.BatchNorm2d(channels)
        self.classifier = nn.Linear(channels, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        features = torch.relu(self.normalization(self.features(inputs)))
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
        return cast(Tensor, self.classifier(pooled))
