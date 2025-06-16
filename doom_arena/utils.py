from typing import Dict
import os
import sys
import re
import numpy as np
import vizdoom as vzd
import torch
import torch.nn.functional as F
from contextlib import contextmanager

from doom_arena.player import player_setup


CHANNELS = {
    vzd.ScreenFormat.CBCGCR: 3,
    vzd.ScreenFormat.CRCGCB: 3,
    vzd.ScreenFormat.GRAY8: 1,
}


def get_screen_shape(
    screen_format,
    screen_resolution,
    labels: bool = False,
    depth: bool = False,
    automap: bool = False,
):
    ch = CHANNELS[screen_format] + int(labels) + int(depth) + 3 * int(automap)
    match = re.search(r"RES_(\d+)X(\d+)", str(screen_resolution))
    return ch, int(match.group(2)), int(match.group(1))


def get_doom_buttons(player_cfg):
    g = vzd.DoomGame()
    g = player_setup(g, player_cfg)
    return g.get_available_buttons()


@contextmanager
def suppress_stdout(verbose):
    if not verbose:
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
    else:
        yield


def to_tensor(raw: Dict[str, np.ndarray], device="cuda"):
    def _to_tensor(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(device).float()
        return x
    return {k: _to_tensor(v) for k, v in raw.items()}


def resize(raw: Dict[str, torch.Tensor], device="cuda"):
    def _resize(x):
        if x.ndim < 4:
            x = x.unsqueeze(0).to(device).float()  # Convert to float32
        else:
            x = x.to(device).float()  # Convert to float32
        x = F.interpolate(x, (128, 128), mode='bilinear', align_corners=False)
        return x.squeeze(0)
    return {k: _resize(v) for k, v in raw.items()}


def minmax(raw: Dict[str, torch.Tensor]):
    def _minmax(x, key):
        if key == "screen":
            # rgb
            x = x / 255
        else:
            # other
            x = x / (x.max() + 1e-8)
        return torch.nan_to_num(x)

    return {k: _minmax(v, k) for k, v in raw.items()}


def cat_dict(raw: Dict[str, torch.Tensor], device="cuda"):
    buffers = ["screen", "labels", "depth", "automap"]
    tensors = [raw[k].to(device) for k in buffers if k in raw]
    return torch.cat(tensors, dim=0)
