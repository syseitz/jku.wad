from time import time
import tqdm
from torchvision import transforms
import torch
import numpy as np
from torch.nn import functional as F

from arena import VizdoomMPEnv


def resize(x):
    # batch dimension for interpolation
    if x.ndim < 4:
        x = x.unsqueeze(0)
    return F.interpolate(x, (128, 128))


def minmax(x):
    return x / 255.0


def to_tensor(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x


ITERATIONS = 12_000

if __name__ == "__main__":
    frame_transform = transforms.Compose([to_tensor, minmax, resize])
    game = VizdoomMPEnv(
        num_players=2,
        num_bots=0,
        ticrate=5000,
        episode_timeout=ITERATIONS,
        player_transform=frame_transform,
    )

    result = game.reset()
    done = False

    n_resets = 0

    start = time()

    for i in tqdm.trange(ITERATIONS, leave=False):
        if done:
            result = game.reset()
            done = False
            n_resets += 1

        result = game.step(game.action_space.sample())
        done = result[2]

    end = time()
    t = end - start
    print("Results:")
    print("time:", round(t, 3), "s")
    print("fps: ", round(ITERATIONS / t, 2))
    print("n resets: ", n_resets)
