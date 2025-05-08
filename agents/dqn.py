from typing import Dict

import random
import torch
from torch import nn
from einops import rearrange
from gym import Env

from agents.utils import Downsample, ResidualBlock


class DQN(nn.Module):
    def __init__(
        self, space: int = 2, input_dim: int = 3, action_space: int = 8, dim: int = 128
    ):
        super().__init__()

        if space == 2:
            self.embed = nn.Conv2d(input_dim, dim, kernel_size=1)
        if space == 3:
            self.embed = nn.Conv3d(input_dim, dim, kernel_size=1)

        self.resnet = nn.Sequential(
            ResidualBlock(space, dim=dim, kernel_size=3, padding=1),
            Downsample(space, dim=dim, downsample=4),
            ResidualBlock(space, dim=dim, kernel_size=3, padding=1),
            Downsample(space, dim=dim, downsample=4),
            ResidualBlock(space, dim=dim, kernel_size=3, padding=1),
            Downsample(space, dim=dim),
            ResidualBlock(space, dim=dim, kernel_size=3, padding=1),
        )

        self.mlp = nn.Sequential(
            nn.Linear(4**2 * dim, 512), nn.SiLU(), nn.Linear(512, action_space)
        )

    def forward(self, frames: torch.Tensor):
        x = self.embed(frames)
        x = self.resnet(x)
        # meanpool
        x = rearrange(x, "b c ... -> b (c ...)")
        x = self.mlp(x)
        return x


@torch.no_grad
def epsilon_greedy(
    env: Env,
    model: nn.Module,
    state: torch.Tensor,
    epsilon: float,
    device: torch.device,
    dtype: torch.dtype,
):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state = state.to(device, dtype=dtype).unsqueeze(0)
        return model(state).argmax().item()
