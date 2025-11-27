"""
MSRF-NAFNet Model Architecture (copied for GUI)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        y = self.weight.view(1, C, 1, 1) * y + self.bias.view(1, C, 1, 1)
        return y


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class MultiScaleConv(nn.Module):
    def __init__(self, channels, scales=[1, 3, 5, 7]):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=k, padding=k//2, groups=channels)
            for k in scales
        ])
        self.fusion = nn.Conv2d(channels * len(scales), channels, 1)
        
    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        fused = torch.cat(outputs, dim=1)
        return self.fusion(fused)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(attention))
        return x * attention


class TextureAwareBlock(nn.Module):
    def __init__(self, channels, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_channels = channels * dw_expand
        
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.multi_scale = MultiScaleConv(dw_channels)
        self.sg = SimpleGate()
        self.conv2 = nn.Conv2d(dw_channels // 2, channels, 1)
        
        self.channel_attn = ChannelAttention(channels)
        self.spatial_attn = SpatialAttention()
        
        ffn_channels = channels * ffn_expand
        self.ffn_conv1 = nn.Conv2d(channels, ffn_channels, 1)
        self.ffn_conv2 = nn.Conv2d(ffn_channels // 2, channels, 1)
        
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.multi_scale(x)
        x = self.sg(x)
        x = self.conv2(x)
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        x = residual + x * self.beta
        
        residual = x
        x = self.norm2(x)
        x = self.ffn_conv1(x)
        x = self.sg(x)
        x = self.ffn_conv2(x)
        x = residual + x * self.gamma
        
        return x


class ContextAggregation(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.project_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v)
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)
        
        return out


class DownSample(nn.Module):
    def __init__(self, in_channels, scale=2):
        super().__init__()
        self.down = nn.Sequential(
            nn.PixelUnshuffle(scale),
            nn.Conv2d(in_channels * scale * scale, in_channels * 2, 1)
        )
        
    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, scale=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * scale * scale // 2, 1),
            nn.PixelShuffle(scale)
        )
        
    def forward(self, x):
        return self.up(x)


class MSRFNAFNet(nn.Module):
    def __init__(
        self,
        img_channels=3,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2]
    ):
        super().__init__()
        
        self.intro = nn.Conv2d(img_channels, width, 3, padding=1)
        self.ending = nn.Conv2d(width, img_channels, 3, padding=1)
        
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[TextureAwareBlock(chan) for _ in range(num)])
            )
            self.downs.append(DownSample(chan))
            chan = chan * 2
        
        self.middle_blks = nn.ModuleList()
        for _ in range(middle_blk_num):
            self.middle_blks.append(TextureAwareBlock(chan))
        
        self.context_agg = nn.ModuleList([
            ContextAggregation(chan) for _ in range(3)
        ])
        
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        for num in dec_blk_nums:
            self.ups.append(UpSample(chan))
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[TextureAwareBlock(chan) for _ in range(num)])
            )
        
        self.skip_fusions = nn.ModuleList()
        chan = width
        for _ in enc_blk_nums:
            self.skip_fusions.append(nn.Conv2d(chan * 2, chan, 1))
            chan = chan * 2
    
    def forward(self, x):
        inp_residual = x
        x = self.intro(x)
        
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        
        for i, blk in enumerate(self.middle_blks):
            x = blk(x)
            if i % 4 == 0 and i // 4 < len(self.context_agg):
                x = x + self.context_agg[i // 4](x)
        
        for decoder, up, enc_skip, fusion in zip(
            self.decoders, self.ups, reversed(encs), reversed(self.skip_fusions)
        ):
            x = up(x)
            x = fusion(torch.cat([x, enc_skip], dim=1))
            x = decoder(x)
        
        x = self.ending(x)
        x = x + inp_residual
        
        return x


def create_msrf_nafnet_s():
    return MSRFNAFNet(
        img_channels=3,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2]
    )


def create_msrf_nafnet_m():
    return MSRFNAFNet(
        img_channels=3,
        width=48,
        middle_blk_num=16,
        enc_blk_nums=[2, 3, 4, 8],
        dec_blk_nums=[2, 3, 4, 2]
    )


def create_msrf_nafnet_l():
    return MSRFNAFNet(
        img_channels=3,
        width=64,
        middle_blk_num=20,
        enc_blk_nums=[3, 3, 6, 12],
        dec_blk_nums=[3, 3, 6, 3]
    )
