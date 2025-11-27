import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, channels, drop_path=0.0):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels * 2, kernel_size=1, padding=0, bias=True)
        self.dwconv = nn.Conv2d(
            channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2, bias=True
        )
        self.simple_gate = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * 2, kernel_size=1, padding=0, bias=True)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.simple_gate(x)
        x = self.sca(x) * x
        x = self.conv2(x)
        x = residual + self.drop_path(x) * self.beta

        residual2 = x
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.simple_gate(x)
        x = self.conv4(x)
        x = residual2 + self.drop_path(x) * self.gamma
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


class NAFNetSmall(nn.Module):
    def __init__(
        self,
        width=32,
        enc_blocks=(2, 2, 4, 8),
        middle_blocks=8,
        dec_blocks=(2, 2, 2, 2),
        num_channels=3,
    ):
        super().__init__()
        self.intro = nn.Conv2d(num_channels, width, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, num_channels, kernel_size=3, padding=1, bias=True)

        enc_channels = [width, width * 2, width * 4, width * 8]
        dec_channels = enc_channels[::-1]

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, num_blocks in enumerate(enc_blocks):
            blocks = [NAFBlock(enc_channels[i]) for _ in range(num_blocks)]
            self.encoders.append(nn.Sequential(*blocks))
            if i < len(enc_blocks) - 1:
                self.downs.append(
                    nn.Conv2d(enc_channels[i], enc_channels[i + 1], kernel_size=2, stride=2, bias=True)
                )

        self.mid = nn.Sequential(*[NAFBlock(enc_channels[-1]) for _ in range(middle_blocks)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i, num_blocks in enumerate(dec_blocks):
            blocks = [NAFBlock(dec_channels[i]) for _ in range(num_blocks)]
            self.decoders.append(nn.Sequential(*blocks))
            if i < len(dec_blocks) - 1:
                self.ups.append(
                    nn.ConvTranspose2d(dec_channels[i], dec_channels[i + 1], kernel_size=2, stride=2, bias=True)
                )

        self.fusions = nn.ModuleList()
        for i in range(len(dec_channels) - 1):
            fused_channels = dec_channels[i + 1] + enc_channels[len(enc_channels) - 2 - i]
            self.fusions.append(nn.Conv2d(fused_channels, dec_channels[i + 1], kernel_size=1, bias=True))

    def forward(self, x):
        x = self.intro(x)

        enc_feats = []
        out = x
        for i, encoder in enumerate(self.encoders):
            out = encoder(out)
            enc_feats.append(out)
            if i < len(self.downs):
                out = self.downs[i](out)

        out = self.mid(out)

        for i, decoder in enumerate(self.decoders):
            out = decoder(out)
            if i < len(self.ups):
                out = self.ups[i](out)
                skip = enc_feats[len(enc_feats) - 2 - i]
                out = torch.cat([out, skip], dim=1)
                out = self.fusions[i](out)

        out = self.ending(out)
        return out  # residual prediction
