import random
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from gym import Env


class Downsample(nn.Module):
    def __init__(self, dim: int, downsample: int = 2):
        super().__init__()

        self.conv = nn.Conv2d(dim, dim, downsample, downsample, bias=False)

    def forward(self, x: torch.Tensor):
        return F.silu(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        act_fn: nn.Module = nn.SiLU,
        depth: int = 2,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        convs = []
        for d in range(depth):
            conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(dim),
                act_fn(),
                nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(dim),
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


class DQN(nn.Module):
    def __init__(self, input_dim: int = 3, action_space: int = 6, dim: int = 128):
        super().__init__()

        self.embed = nn.Conv2d(input_dim, dim, kernel_size=1)
        self.resnet = nn.Sequential(
            ResidualBlock(dim, kernel_size=3, padding="same"),
            Downsample(dim),
            ResidualBlock(dim, kernel_size=3, padding="same"),
            Downsample(dim),
            ResidualBlock(dim, kernel_size=3, padding="same"),
            Downsample(dim),
            ResidualBlock(dim, kernel_size=3, padding="same"),
            Downsample(dim),
            ResidualBlock(dim, kernel_size=3, padding="same"),
            Downsample(dim),
            ResidualBlock(dim, kernel_size=3, padding="same"),
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim, 512), nn.SiLU(), nn.Linear(512, action_space)
        )

    def forward(self, x):
        x = self.embed(x)
        x = self.resnet(x)
        # meanpool
        x = rearrange(x, "b c h w -> b c (h w)").mean(-1)
        x = self.mlp(x)
        return x


@torch.no_grad
def epsilon_greedy(
    env: Env,
    model: nn.Module,
    state: torch.Tensor,
    epsilon: float,
    device: torch.device,
):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state = state.to(device)
        return model(state).argmax().item()
