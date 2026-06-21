from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NGPSConfig


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SCA(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(self.avg_pool(x))


class NAFBlock(nn.Module):
    def __init__(self, channels: int, dw_expand: int = 2, ff_expand: int = 2) -> None:
        super().__init__()
        dw_channels = channels * dw_expand
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, kernel_size=3, padding=1, groups=dw_channels, bias=True)
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1, bias=True)
        self.sca = SCA(dw_channels // 2)
        self.sg = SimpleGate()

        ff_channels = channels * ff_expand
        self.conv4 = nn.Conv2d(channels, ff_channels, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ff_channels // 2, channels, kernel_size=1, bias=True)

        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca(x)
        x = self.conv3(x)
        residual = residual + x * self.beta

        x = self.norm2(residual)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return residual + x * self.gamma


class NAFNet(nn.Module):
    def __init__(
        self,
        img_channel: int = 1,
        width: int = 32,
        middle_blk_num: int = 8,
        enc_blk_nums: Tuple[int, ...] = (2, 2, 4, 8),
        dec_blk_nums: Tuple[int, ...] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        self.padder_size = 2 ** len(enc_blk_nums)
        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, kernel_size=3, padding=1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        channels = width
        for num_blocks in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(num_blocks)]))
            self.downs.append(nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2))
            channels *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(channels) for _ in range(middle_blk_num)])

        for num_blocks in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False), nn.PixelShuffle(2)))
            channels //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(num_blocks)]))

    def _pad_to_stride(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        _, _, height, width = x.shape
        pad_h = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - width % self.padder_size) % self.padder_size
        if pad_h == 0 and pad_w == 0:
            return x, height, width
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), height, width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, height, width = self._pad_to_stride(x)
        x = self.intro(x)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x)
            x = decoder(x + skip)
        x = self.ending(x)
        return x[:, :, :height, :width]


def build_nafnet(cfg: NGPSConfig = NGPSConfig()) -> NAFNet:
    return NAFNet(
        img_channel=1,
        width=cfg.nafnet_width,
        middle_blk_num=cfg.nafnet_middle_blocks,
        enc_blk_nums=cfg.nafnet_enc_blocks,
        dec_blk_nums=cfg.nafnet_dec_blocks,
    )
