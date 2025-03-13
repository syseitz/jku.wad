import random
import torch
from torch import nn
import torch.nn.functional as F
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
            ResidualBlock(space, dim=dim, kernel_size=3, padding="same"),
            Downsample(space, dim=dim),
            ResidualBlock(space, dim=dim, kernel_size=3, padding="same"),
            Downsample(space, dim=dim),
            ResidualBlock(space, dim=dim, kernel_size=3, padding="same"),
            Downsample(space, dim=dim),
            ResidualBlock(space, dim=dim, kernel_size=3, padding="same"),
            Downsample(space, dim=dim),
            ResidualBlock(space, dim=dim, kernel_size=3, padding="same"),
            Downsample(space, dim=dim),
            ResidualBlock(space, dim=dim, kernel_size=3, padding="same"),
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim, 512), nn.SiLU(), nn.Linear(512, action_space)
        )

    def forward(self, x):
        x = self.embed(x)
        x = self.resnet(x)
        # meanpool
        x = rearrange(x, "b c ... -> b c (...)").mean(-1)
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
