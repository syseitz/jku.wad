import os
import sys
import re
import vizdoom as vd
from contextlib import contextmanager

from arena.player import player_setup


CHANNELS = {
    vd.ScreenFormat.CBCGCR: 3,
    vd.ScreenFormat.GRAY8: 3,
}


def get_screen_shape(screen_format, screen_resolution):
    ch = CHANNELS[screen_format]
    match = re.search(r"RES_(\d+)X(\d+)", str(screen_resolution))
    return ch, int(match.group(1)), int(match.group(2))


def get_action_dim(player_cfg):
    g = vd.DoomGame()
    g = player_setup(g, player_cfg)
    return len(g.get_available_buttons())


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
