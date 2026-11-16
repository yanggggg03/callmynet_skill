"""Two measured inputs; bounded lengths; independent forward-response surrogate."""

import math

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )

    def forward(self, x):
        return x + self.net(x)


class ResidualDesigner(nn.Module):
    is_flow = False

    def __init__(self, width=128, blocks=4, fourier_features=0, fourier_scale=1.0):
        super().__init__()
        self.register_buffer("B", torch.randn(2, fourier_features) * fourier_scale)
        self.trunk = nn.Sequential(
            nn.Linear(2 + 2 * fourier_features, width),
            nn.SiLU(),
            *[ResidualBlock(width) for _ in range(blocks)],
            nn.LayerNorm(width)
        )
        self.length = nn.Linear(width, 2)
        self.uncertainty = nn.Linear(width, 2)

    def forward(self, x):
        if self.B.shape[1]:
            angle = 2 * torch.pi * (x @ self.B)
            x = torch.cat([x, angle.sin(), angle.cos()], -1)
        h = self.trunk(x)
        return torch.sigmoid(self.length(h)), 0.01 + 0.29 * torch.sigmoid(
            self.uncertainty(h)
        )


class Coupling(nn.Module):
    def __init__(self, mask, width):
        super().__init__()
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(width + 2, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 4),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, c, reverse=False):
        shift, scale = self.net(torch.cat([x * self.mask, c], -1)).chunk(2, -1)
        shift = shift * (1 - self.mask)
        scale = 0.75 * torch.tanh(scale) * (1 - self.mask)
        if reverse:
            return (x - shift) * torch.exp(-scale), -scale.sum(-1)
        return x * torch.exp(scale) + shift, scale.sum(-1)


class ConditionalFlow(nn.Module):
    is_flow = True

    def __init__(self, width=128, layers=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU()
        )
        self.flow = nn.ModuleList(
            [Coupling([1, 0] if i % 2 == 0 else [0, 1], width) for i in range(layers)]
        )

    def log_prob(self, x, target):
        # Density on normalized actuator interval; no false half-wave wrap assumption.
        y = target.clamp(1e-4, 1 - 1e-4)
        u = torch.logit(y)
        ld = -torch.log(y * (1 - y)).sum(-1)
        c = self.encoder(x)
        for layer in self.flow:
            u, det = layer(u, c)
            ld = ld + det
        return -0.5 * (u.square() + math.log(2 * math.pi)).sum(-1) + ld

    def sample(self, x, count=32, generator=None):
        encoded = self.encoder(x)
        c = encoded[:, None, :].expand(-1, count, -1).reshape(-1, encoded.shape[-1])
        z = torch.randn(len(x) * count, 2, device=x.device, generator=generator)
        for layer in reversed(self.flow):
            z, _ = layer(z, c, reverse=True)
        return torch.sigmoid(z).reshape(len(x), count, 2)


class ResponseSurrogate(nn.Module):
    """Predict measured complex reflection after a candidate action, not its truth."""

    def __init__(self, width=96, blocks=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, width),
            nn.SiLU(),
            *[ResidualBlock(width, 0.0) for _ in range(blocks)],
            nn.Linear(width, 2)
        )

    def forward(self, condition, lengths):
        raw = self.net(torch.cat([condition, lengths], -1))
        # Radial squash guarantees |Gamma| <= 1 (componentwise tanh would not).
        return raw / torch.sqrt(1 + raw.square().sum(-1, keepdim=True))


def build(name, config):
    if name == "conditional_flow":
        return ConditionalFlow(config["width"], config["layers"])
    return ResidualDesigner(
        config["width"],
        config["blocks"],
        config["fourier_features"] if name == "fourier_resmlp" else 0,
        config["fourier_scale"],
    )
