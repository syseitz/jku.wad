import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict


def stack_dict(x):
    return np.concat(list(x.values()), 0)


def to_tensor(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x


def resize(x: torch.Tensor):
    if x.ndim < 4:
        x = x.unsqueeze(0)
    x = F.interpolate(x, (128, 128))
    return x.squeeze(0)


def minmax(x: torch.Tensor):
    # rgb
    x[:3] = x[:3] / 255
    if x.shape[0] > 3:
        for c in range(3, x.shape[0]):
            x[c] = x[c] / (x[c].max() + 1e-8)
    return torch.nan_to_num(x)


class Downsample(nn.Module):
    def __init__(self, space: int, dim: int, downsample: int = 2):
        super().__init__()

        if space == 2:
            Conv = nn.Conv2d
        else:
            Conv = nn.Conv3d
            downsample = (1, downsample, downsample)

        self.conv = Conv(dim, dim, downsample, downsample, bias=False)

    def forward(self, x: torch.Tensor):
        return F.silu(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        space: int,
        dim: int,
        act_fn: nn.Module = nn.SiLU,
        depth: int = 2,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        Conv = nn.Conv2d if space == 2 else nn.Conv3d
        convs = []
        for d in range(depth):
            conv = nn.Sequential(
                Conv(dim, dim, kernel_size=kernel_size, padding=padding),
                act_fn(),
                Conv(dim, dim, kernel_size=kernel_size, padding=padding),
            )
            convs.append(conv)
            if d < depth - 1:
                convs.append(act_fn())

        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        residual = x
        for conv in self.convs:
            x = conv(x) + residual
            residual = x
        return x


@torch.no_grad()
def update_ema(ema_model, model, decay: float = 0.995):
    ema_params = OrderedDict(ema_model.named_parameters())
    if hasattr(model, "module"):
        model_params = OrderedDict(model.module.named_parameters())
    else:
        model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
