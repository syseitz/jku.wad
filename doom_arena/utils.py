import os
import sys
import re
import vizdoom as vzd
from contextlib import contextmanager

from doom_arena.player import player_setup


CHANNELS = {
    vzd.ScreenFormat.CBCGCR: 3,
    vzd.ScreenFormat.GRAY8: 3,
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
