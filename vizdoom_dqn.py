import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from collections import deque

from doom_arena import VizdoomMPEnv
from doom_arena.player import ObsBuffer

from agents.dqn import DQN, epsilon_greedy


def stack_dict(x):
    return np.concat(list(x.values()), 0)


def to_tensor(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x


def resize(x):
    return F.interpolate(x.unsqueeze(0), (128, 128))


def minmax(x):
    # channelwise minmax (preserves different buffers as well)
    x_max = x.view(x.shape[0], x.shape[1], -1).max(-1)[0][..., *[None] * (x.ndim - 2)]
    x = x / (x_max + 1e-8)
    return torch.nan_to_num(x)


device = "cuda"

GAMMA = 0.95
EPOCHS = 10
EPISODES = 100
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 10000
LEARNING_RATE = 1e-4
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 50000

if __name__ == "__main__":
    frame_transform = transforms.Compose([stack_dict, to_tensor, resize, minmax])
    env = VizdoomMPEnv(
        num_players=1,
        num_bots=8,
        doom_map="map03",
        extra_state=[ObsBuffer.LABELS],
        episode_timeout=2000,
        player_transform=frame_transform,
    )

    dqn = DQN(
        input_dim=env.observation_space.shape[0],
        action_space=env.action_space.n,
        dim=128,
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
        dqn.eval()

        while not done:
            act = epsilon_greedy(env, dqn, obs, epsilon, device)
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
            dqn.train()
            for e in range(EPOCHS):
                batch = random.sample(replay_buffer, BATCH_SIZE)
                obs, actions, rewards, next_obs, dones = zip(*batch)

                obs = torch.cat(obs).to(device, dtype=torch.float32)
                next_obs = torch.cat(next_obs).to(device, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.long).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)

                q_values = dqn(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q_values = dqn(next_obs).max(1).values
                    target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

                loss = F.mse_loss(q_values, target_q_values)
                q_loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update epsilon
            epsilon = max(EPSILON_END, EPSILON_START - steps_done / EPSILON_DECAY)

        avg_reward = np.mean(reward_list[-10:])
        print(
            f"Episode {episode + 1}/{EPISODES}, "
            f"steps: {ep_step}, epsilon: {epsilon:.2f}"
        )
        print(f"\tReturn: {ep_return:.2f}, avg reward (last 10): {avg_reward:.2f}")
        if len(q_loss_list) > 0:
            avg_q_loss = np.mean(q_loss_list[-10:])
            print(f"\tAvg Q-loss: {avg_q_loss:.4f}")
        print()

    torch.save(dqn.state_dict(), "dqn.pth")
