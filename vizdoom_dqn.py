import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from collections import deque
from copy import deepcopy

from doom_arena import VizdoomMPEnv
from doom_arena.player import ObsBuffer

from agents.dqn import DQN, epsilon_greedy
from agents.utils import stack_dict, to_tensor, resize, minmax, update_ema


device = "cuda"

N_EPOCHS = 50
DTYPE = torch.float32
N_STACK_FRAMES = 1
GAMMA = 0.95
EPISODES = 500
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000
LEARNING_RATE = 1e-4
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995

if __name__ == "__main__":
    frame_transform = transforms.Compose([stack_dict, to_tensor, resize, minmax])
    env = VizdoomMPEnv(
        num_players=1,
        num_bots=16,
        bot_skill=0,
        doom_map="TRNM",
        extra_state=[ObsBuffer.LABELS, ObsBuffer.DEPTH],
        episode_timeout=120 * 35,
        n_stack_frames=N_STACK_FRAMES,
        player_transform=[frame_transform]
    )

    dqn = DQN(
        space=3 if N_STACK_FRAMES > 1 else 2,
        input_dim=env.observation_space.shape[0],
        action_space=env.action_space.n,
        dim=64,
    ).to(device, dtype=DTYPE)

    dqn_tgt = deepcopy(dqn).to(device, dtype=DTYPE)
    for p in dqn_tgt.parameters():
        p.requires_grad = False
    update_ema(dqn_tgt, dqn, decay=0)
    dqn_tgt.eval()

    optimizer = optim.Adam(dqn.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.995)

    print(f"Parameters: {sum(p.numel() for p in dqn.parameters()) / 1e6:.1f}M")
    # Replay buffer
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    epsilon = EPSILON_START
    steps_done = 0
    best_eval_return = -np.inf
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
            act = epsilon_greedy(env, dqn, obs, epsilon, device, DTYPE)
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
        dqn.train()
        for _ in range(N_EPOCHS):
            batch = random.sample(replay_buffer, BATCH_SIZE)
            obs, actions, rewards, next_obs, dones = zip(*batch)

            obs = torch.stack(obs, 0).to(device, dtype=DTYPE)
            next_obs = torch.stack(next_obs, 0).to(device, dtype=DTYPE)
            actions = torch.tensor(actions, dtype=torch.long).to(device)
            rewards = torch.tensor(rewards, dtype=DTYPE).to(device)
            dones = torch.tensor(dones, dtype=DTYPE).to(device)

            q_values = dqn(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_values = dqn_tgt(next_obs).max(1).values
                target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

            loss = F.mse_loss(q_values, target_q_values)
            q_loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Update epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        update_ema(dqn_tgt, dqn)

        # evaluate
        done = False
        obs = env.reset()
        eval_return = 0
        while not done:
            obs = obs[0].to(device, dtype=DTYPE)
            with torch.no_grad():
                act = dqn(obs.unsqueeze(0)).argmax().item()
            obs, rwd, done, info = env.step(act)
            eval_return += rwd[0]

        if eval_return > best_eval_return:
            best_eval_return = eval_return
            best_dqn = deepcopy(dqn)

        print(
            f"Episode {episode + 1}/{EPISODES}, "
            f"steps: {ep_step}, epsilon: {epsilon:.2f}"
        )
        print(f"\tReturn: {ep_return:.2f}, eval return: {eval_return:.2f}")
        if len(q_loss_list) > 0:
            avg_q_loss = np.mean(q_loss_list[-10:])
            print(f"\tAvg Q-loss: {avg_q_loss:.4f}")
        print()

    print(f"Best return: {best_eval_return:.2f}")

    torch.save(dqn.state_dict(), "dqn.pth")
    torch.save(best_dqn.state_dict(), "best_dqn.pth")
    torch.save(dqn_tgt.state_dict(), "dqn_tgt.pth")
