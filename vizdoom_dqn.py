import random
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from einops import rearrange
from collections import deque

from arena import VizdoomMPEnv
from arena.player import ObsBuffer


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
def epsilon_greedy(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state = state.to(device)
        return dqn(state).argmax().item()


def stack_dict(x):
    return np.concat([v for v in x.values()], 1)


def to_tensor(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x


def resize(x):
    # batch dimension for interpolation
    if x.ndim < 4:
        x = x.unsqueeze(0)
    return F.interpolate(x, (128, 128))


def minmax(x):
    # channelwise minmax (preserves different buffers as well)
    x_max = x.view(x.shape[0], x.shape[1], -1).max(-1)[0][..., *[None] * (x.ndim - 2)]
    x = x / (x_max + 1e-8)
    return torch.nan_to_num(x)


device = "cuda"

GAMMA = 0.95
EPOCHS = 10
EPISODES = 100
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 10000
LEARNING_RATE = 1e-4
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 20000


if __name__ == "__main__":
    frame_transform = transforms.Compose([stack_dict, to_tensor, minmax, resize])
    env = VizdoomMPEnv(
        num_players=1,
        num_bots=4,
        extra_state=[ObsBuffer.LABELS],
        n_stack_frames=1,
        episode_timeout=1000,
        player_transform=frame_transform,
    )

    dqn = DQN(
        input_dim=env.observation_space.shape[0],
        action_space=env.action_space.n,
        dim=64,
    ).to(device)
    optimizer = optim.Adam(dqn.parameters(), lr=LEARNING_RATE)

    print(f"Parameters: {sum(p.numel() for p in dqn.parameters()) / 1e6:.1f}M")
    # Replay buffer
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    epsilon = EPSILON_START
    steps_done = 0
    q_loss_list = []
    reward_list = []

    for episode in range(EPISODES):
        ep_return = 0.0
        ep_step = 0
        obs = env.reset()
        obs = obs[0]  # Single player

        done = False
        while not done:
            act = epsilon_greedy(obs, epsilon)
            next_obs, rwd, done, info = env.step(act)

            # Single player
            rwd = rwd[0]
            next_obs = next_obs[0]
            # Store in replay buffer
            replay_buffer.append((obs, act, rwd, next_obs, done))
            ep_return += rwd
            steps_done += 1
            ep_step += 1
            obs = next_obs

        reward_list.append(ep_return)

        # train if buffer has enough samples
        if len(replay_buffer) > REPLAY_BUFFER_SIZE // 4:
            for e in range(EPOCHS):
                batch = random.sample(replay_buffer, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.cat(states).to(device, dtype=torch.float32)
                next_states = torch.cat(next_states, 0).to(device, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.long).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)

                q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q_values = dqn(next_states).max(1).values
                    target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

                loss = F.mse_loss(q_values, target_q_values)
                q_loss_list.append(loss.item())  # Store loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update epsilon
            epsilon = max(EPSILON_END, EPSILON_START - steps_done / EPSILON_DECAY)

        avg_reward = np.mean(reward_list[-10:])
        print(
            f"Episode {episode + 1}/{EPISODES},"
            f"steps: {ep_step}, epsilon: {epsilon:.2f}"
        )
        print(f"\tReturn: {ep_return:.2f}, avg reward (last 10): {avg_reward:.2f}")
        if len(q_loss_list) > 0:
            avg_q_loss = np.mean(q_loss_list[-10:])
            print(f"\tAvg Q-loss: {avg_q_loss:.4f}")
        print()
