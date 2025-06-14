from typing import Callable, Optional, Sequence, List, Union, Dict
from concurrent import futures


import argparse
import functools
import json
import os
import numpy as np
import random
import re

import torch
from torch.distributions.categorical import Categorical

import onnx
from onnx2pytorch import ConvertModel

from gym.spaces import Tuple as GymTuple
from copy import deepcopy
from collections import deque
from portpicker import pick_unused_port
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from warnings import warn
import copy
import enum

import vizdoom as vzd
from gym import Env
from torchvision import transforms
from gym.spaces import Box, MultiDiscrete, Discrete


################################## grading reward ######################################


class VizDoomReward:
    def __init__(self, num_players: int):
        self.num_players = num_players
        self.reset()

    def reset(self):
        self._step = 0
        self.cumulated_frag = np.zeros(self.num_players)

    def __call__(
        self,
        vizdoom_reward: float,
        game_var: Dict[str, float],
        game_var_old: Dict[str, float],
        player_id: int,
    ) -> Sequence:
        self._step += 1
        _ = player_id, vizdoom_reward
        rwd_hit = 2.0 * (game_var["HITCOUNT"] - game_var_old["HITCOUNT"])
        rwd_hit_taken = -0.1 * (game_var["HITS_TAKEN"] - game_var_old["HITS_TAKEN"])
        rwd_frag = 100.0 * (game_var["FRAGCOUNT"] - game_var_old["FRAGCOUNT"])
        # rwd_health = (game_var["HEALTH"] - game_var_old["HEALTH"])
        # rwd_dead = -10.0 if (game_var["DEAD"] - game_var_old["DEAD"]) > 0 else 0.0

        return (
            # vizdoom_reward,
            rwd_hit,
            rwd_hit_taken,
            rwd_frag,
            # rwd_health,
            # rwd_dead,
        )


################################# doom arena utils #####################################


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


def to_tensor(raw: Dict[str, np.ndarray]):
    def _to_tensor(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x

    return {k: _to_tensor(v) for k, v in raw.items()}


def resize(raw: Dict[str, torch.Tensor]):
    def _resize(x):
        if x.ndim < 4:
            x = x.unsqueeze(0)
        x = F.interpolate(x, (128, 128))
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


def cat_dict(raw: Dict[str, torch.Tensor]):
    # concat along channels
    buffers = ["screen", "labels", "depth", "automap"]
    return torch.cat([raw[k] for k in buffers if k in raw], 0)


############################# doom arena player config #################################


class PlayerHostConfig(object):
    def __init__(self, port, num_players=2):
        self.num_players = num_players
        self.port = port
        print("Host {}".format(self.port))


class PlayerJoinConfig(object):
    def __init__(self, port):
        self.join_ip = "127.0.0.1"
        self.port = port
        print("Player {}".format(self.port))


class PlayerConfig(object):
    def __init__(self):
        self.config_path = None
        self.player_mode = None
        self.screen_resolution = None
        self.screen_format = None
        self.ticrate = None
        self.episode_timeout = None
        # custom character
        self.name = None
        self.colorset = None
        self.crosshair = None
        self.hud = "full"
        # observation settings
        self.transform = None
        self.use_depth = False
        self.use_labels = False
        self.use_automap = False
        # game variables
        self.available_game_variables = [
            vzd.GameVariable.HEALTH,
            vzd.GameVariable.AMMO3,
            # additional
            vzd.GameVariable.FRAGCOUNT,
            vzd.GameVariable.ARMOR,
            vzd.GameVariable.HITCOUNT,
            vzd.GameVariable.HITS_TAKEN,
            vzd.GameVariable.DEAD,
            vzd.GameVariable.DEATHCOUNT,
            vzd.GameVariable.DAMAGECOUNT,
            vzd.GameVariable.DAMAGE_TAKEN,
            vzd.GameVariable.KILLCOUNT,
            vzd.GameVariable.SELECTED_WEAPON,
            vzd.GameVariable.SELECTED_WEAPON_AMMO,
            vzd.GameVariable.POSITION_X,
            vzd.GameVariable.POSITION_Y,
        ]
        # game buttons
        # NOTE: order is impotant
        self.available_buttons = [
            vzd.Button.MOVE_FORWARD,
            vzd.Button.ATTACK,
            vzd.Button.MOVE_LEFT,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.TURN_LEFT,
            vzd.Button.TURN_RIGHT,
            vzd.Button.JUMP,
            # vzd.Button.USE,
            # vzd.Button.MOVE_BACKWARD,
        ]

        # game map
        self.doom_map = None
        self.doom_skill = 1
        self.bot_skill = 0

        self.n_stack_frames = 1
        self.num_bots = 0
        self.respawns = False

        self.host_cfg = None
        self.join_cfg = None


def player_host_setup(game, host_config):
    game.add_game_args(
        " ".join(
            [
                "-host {}".format(host_config.num_players),
                "-port {}".format(host_config.port),
                "-netmode 0",
                "-deathmatch",
                "+sv_spawnfarthest 1",
                "+viz_nocheat 0",
            ]
        )
    )
    return game


def player_join_setup(game, join_config):
    game.add_game_args(
        " ".join(
            [
                "-join {}".format(join_config.join_ip),
                "-port {}".format(join_config.port),
            ]
        )
    )
    return game


def player_setup(game, player_config: PlayerConfig):
    if player_config.config_path is not None:
        game.load_config(player_config.config_path)

    if player_config.player_mode is not None:
        game.set_mode(player_config.player_mode)
    if player_config.screen_resolution is not None:
        game.set_screen_resolution(player_config.screen_resolution)
    if player_config.screen_format is not None:
        game.set_screen_format(player_config.screen_format)
    if player_config.ticrate is not None:
        game.set_ticrate(player_config.ticrate)
    if player_config.episode_timeout is not None:
        game.set_episode_timeout(player_config.episode_timeout)
    if player_config.episode_timeout is not None:
        game.set_episode_timeout(player_config.episode_timeout)

    if player_config.doom_map is not None:
        game.set_doom_map(player_config.doom_map)

    if player_config.doom_skill is not None:
        game.set_doom_skill(player_config.doom_skill)

    # custom game variables
    if player_config.available_game_variables is not None:
        game.set_available_game_variables(player_config.available_game_variables)

    # custom game buttons
    if player_config.available_buttons is not None:
        game.set_available_buttons(player_config.available_buttons)

    # hud
    if player_config.hud != "full":
        if player_config.hud == "minimal":
            game.set_render_minimal_hud(True)
        if player_config.hud in [False, None, "none"]:
            game.set_render_hud(False)

    game.set_render_decals(False)
    game.set_render_messages(False)

    # depth
    if player_config.use_depth:
        game.set_depth_buffer_enabled(True)
    # segmented
    if player_config.use_labels:
        game.set_labels_buffer_enabled(True)
    # automap
    if player_config.use_automap:
        game.set_automap_buffer_enabled(True)
        game.set_automap_mode(vzd.AutomapMode.OBJECTS)
        game.set_automap_rotate(False)
        game.set_automap_render_textures(False)

    game.set_console_enabled(False)

    if player_config.name is not None:
        game.add_game_args("+name {}".format(player_config.name))
    if player_config.colorset is not None:
        game.add_game_args("+colorset {}".format(player_config.colorset))

    if player_config.crosshair is not None:
        game.set_render_crosshair(player_config.crosshair)

    if player_config.respawns:
        # respawn automatically after death
        game.add_game_args("+sv_forcerespawn 1")
        # invulnerable for two second after spawning
        game.add_game_args("+sv_respawnprotect 1")
        # seconds between respawns
        game.add_game_args(f"+viz_respawn_delay 5")
    return game


def mp_game_setup(game, bot_skill: int = 0):
    # respawn items
    game.add_game_args("+altdeath 1")
    # no crouching
    game.add_game_args("+sv_nocrouch 1")
    # no monsters
    game.add_game_args("+sv_nomonsters 1")
    game.add_game_args("+nomonsters 1")
    if bot_skill == 0:
        # easy bots
        game.add_game_args(f"+viz_bots_path {os.getcwd()}/doom_arena/bots/easy.cfg")
    if bot_skill == 1:
        pass
    if bot_skill >= 2:
        # hard bots
        game.add_game_args(f"+viz_bots_path {os.getcwd()}/doom_arena/bots/hard.cfg")
    return game


############################## doom arena multiprocess #################################


class RunParallel(object):
    """Run all funcs in parallel."""

    def __init__(self, timeout=None):
        self._timeout = timeout
        self._executor = None
        self._workers = 0

    def run(self, funcs):
        """Run a set of functions in parallel, returning their results.

        Make sure any function you pass exits with a reasonable timeout. If it
        doesn't return within the timeout or the result is ignored due an exception
        in a separate thread it will continue to stick around until it finishes,
        including blocking process exit.

        Args:
          funcs: An iterable of functions or iterable of args to functools.partial.

        Returns:
          A list of return values with the values matching the order in funcs.

        Raises:
          Propagates the first exception encountered in one of the functions.
        """
        funcs = [f if callable(f) else functools.partial(*f) for f in funcs]
        if len(funcs) == 1:  # Ignore threads if it's not needed.
            return [funcs[0]()]
        if len(funcs) > self._workers:  # Lazy init and grow as needed.
            self.shutdown()
            self._workers = len(funcs)
            self._executor = futures.ThreadPoolExecutor(self._workers)
        futs = [self._executor.submit(f) for f in funcs]
        done, not_done = futures.wait(futs, self._timeout, futures.FIRST_EXCEPTION)
        # Make sure to propagate any exceptions.
        for f in done:
            if not f.cancelled() and f.exception() is not None:
                if not_done:
                    # If there are some calls that haven't finished, cancel and recreate
                    # the thread pool. Otherwise we may have a thread running forever
                    # blocking parallel calls.
                    for nd in not_done:
                        nd.cancel()
                    self.shutdown(False)  # Don't wait, they may be deadlocked.
                raise f.exception()
        # Either done or timed out, so don't wait again.
        return [f.result(timeout=0) for f in futs]

    def shutdown(self, wait=True):
        if self._executor:
            self._executor.shutdown(wait)
            self._executor = None
            self._workers = 0

    def __del__(self):
        self.shutdown()


class ParallelEnv(Env):
    """Vectorize a list of environments as a single environment."""

    def __init__(self, envs):
        self._spectator_envs = [e for e in envs if e.is_spectator]
        self._player_envs = [e for e in envs if not e.is_spectator]

        self.observation_space = GymTuple(
            [e.observation_space for e in self._player_envs]
        )
        self.action_space = GymTuple([e.action_space for e in self._player_envs])

        self._par = RunParallel()

    def reset(self):
        observations = self._par.run((e.reset) for e in self.envs)
        return observations

    def step(self, actions):
        ret = self._par.run((e.step, act) for e, act in zip(self._player_envs, actions))
        observations, rewards, dones, infos = [item for item in zip(*ret)]
        return observations, rewards, dones, {}

    def close(self):
        self._par.run((e.close) for e in self._player_envs)

    @property
    def envs(self):
        return self._spectator_envs + self._player_envs


############################### doom arena main env ####################################


class ObsBuffer(enum.Enum):
    LABELS = "labels"
    DEPTH = "depth"
    AUTOMAP = "automap"


# TODO what does this do?
VIZDOOM_ACTIONS = [
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],  # 0 move fast forward
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 fire
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # 2 move left
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # 3 move right
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # 4 turn left
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # 5 turn right
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 20],  # 6 turn left 20 degree and move forward
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 20],  # 7 turn right 20 degree and move forward
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 8 move forward
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 9 turn 180
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 10 move left
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 11 move right
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 12 turn left
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 13 turn right
]


class PlayerEnv(Env):
    """ViZDoom per player environment."""

    def __init__(
        self,
        cfg: PlayerConfig,
        discrete7: bool = True,
        frame_skip: int = 1,
        seed: int = 1337,
    ):
        self.cfg = cfg
        self.discrete7 = discrete7
        self.transform = cfg.transform
        self.record = False
        self.replay = None
        self.is_spectator = cfg.player_mode == vzd.Mode.SPECTATOR
        self.frame_skip = frame_skip
        # self.seed(seed)
        self.doom_seed = seed
        self.game = None

        self.observation_space = Box(
            low=0,
            high=255,
            dtype=np.uint8,
            shape=get_screen_shape(
                cfg.screen_format,
                cfg.screen_resolution,
                labels=cfg.use_labels,
                depth=cfg.use_depth,
                automap=cfg.use_automap,
            ),
        )

        self._buttons = get_doom_buttons(cfg)
        if discrete7:
            # simplified
            self.action_space = Discrete(len(self._buttons) + 1)  # NOTE: +1 for noop
        else:
            # TODO what does this do?
            # actual button space
            self.action_space = MultiDiscrete([2] * len(self._buttons))

        self._state = None
        self._game_var_list = None
        self._game_vars = {}
        self._game_vars_pre = {}

        self.history = deque(maxlen=60)
        self.frame_stack = deque(maxlen=cfg.n_stack_frames)

    def repeat_action(self, action):
        cnt = 1
        while len(self.history) > cnt + 1 and self.history[-(cnt + 1)] == action:
            cnt += 1
        return cnt

    def adjust_action(self, action):
        # TODO what does this do?
        self.history.append(action)
        is_attacking = VIZDOOM_ACTIONS[1] in list(self.history)[-3:]
        if action in [VIZDOOM_ACTIONS[4], VIZDOOM_ACTIONS[5]]:
            if self.repeat_action(action) > 3:
                action = (
                    VIZDOOM_ACTIONS[6]
                    if action == VIZDOOM_ACTIONS[4]
                    else VIZDOOM_ACTIONS[7]
                )
        if action in [VIZDOOM_ACTIONS[0], VIZDOOM_ACTIONS[2], VIZDOOM_ACTIONS[3]]:
            if self.repeat_action(action) % 16 == 0:
                action = VIZDOOM_ACTIONS[9]
        if action in [
            VIZDOOM_ACTIONS[0],
            VIZDOOM_ACTIONS[2],
            VIZDOOM_ACTIONS[3],
            VIZDOOM_ACTIONS[4],
            VIZDOOM_ACTIONS[5],
        ]:
            if is_attacking:
                if action == VIZDOOM_ACTIONS[0]:
                    action = VIZDOOM_ACTIONS[8]
                if action == VIZDOOM_ACTIONS[2]:
                    action = VIZDOOM_ACTIONS[10]
                if action == VIZDOOM_ACTIONS[3]:
                    action = VIZDOOM_ACTIONS[11]
                if action == VIZDOOM_ACTIONS[4]:
                    action = VIZDOOM_ACTIONS[12]
                if action == VIZDOOM_ACTIONS[5]:
                    action = VIZDOOM_ACTIONS[13]
        if self.game.is_player_dead():
            self.history.clear()
        # convert to c++ type
        action = list(action)
        return action

    def reset(self):
        self._init_game()
        if self.cfg.num_bots > 0:
            self._add_bot()

        self.game.new_episode()
        self._state, obs, _ = self._grab()

        self._record_replay(obs, reset=True)
        self._update_game_vars(reset=True)
        obs = self._update_frame_stack(obs, reset=True)

        # apply (frame) transforms
        if self.transform is not None:
            obs = self.transform(obs)
        return obs

    def step(self, action):
        if isinstance(action, np.ndarray) or isinstance(action, torch.Tensor):
            action = action.tolist()
        if self.discrete7:
            # action = VIZDOOM_ACTIONS[action]
            # onehot
            if action == 0:
                # noop
                action = [0] * self.action_space.n
            else:
                action = [
                    1 if (i + 1) == action else 0 for i in range(self.action_space.n)
                ]
        # action = self.adjust_action(action)
        # vizdoom step
        rwd = self.game.make_action(action, tics=self.frame_skip)
        self._state, obs, done = self._grab()

        self._record_replay(obs)
        self._update_game_vars()
        obs = self._update_frame_stack(obs)

        if done:
            # game over, all obs to white
            game_over_obs = {}
            for k in obs:
                shape = get_screen_shape(
                    self.cfg.screen_format,
                    self.cfg.screen_resolution,
                    labels=(k == "labels"),
                    depth=(k == "depth"),
                    automap=(k == "automap"),
                )
                shape = list(shape)
                if k != "screen":
                    shape[0] = shape[0] - 3  # remove rgb
                game_over_obs[k] = np.ones(shape, dtype=np.uint8) * 255
            obs = game_over_obs

        # apply (frame) transforms
        if self.transform is not None:
            obs = self.transform(obs)
        return obs, rwd, done, {}

    def close(self):
        if self.game:
            self.game.close()

    def _init_game(self):
        self.close()
        game = vzd.DoomGame()
        game = player_setup(game, self.cfg)
        game = mp_game_setup(game, self.cfg.bot_skill)
        if self.cfg.host_cfg is not None:
            game = player_host_setup(game, self.cfg.host_cfg)
        elif self.cfg.join_cfg is not None:
            game = player_join_setup(game, self.cfg.join_cfg)
        else:
            raise ValueError("neither host nor join, error!")
        game.set_window_visible(False)  # NOTE: for devices without screen
        game.set_seed(self.doom_seed)
        game.init()
        self.game = game

        self._game_var_list = self.game.get_available_game_variables()
        self._update_game_vars(reset=True)

    def _grab(self):
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        if done:
            # set observation to black if done
            tmp = np.ndarray(self.observation_space.shape, self.observation_space.dtype)
            obs = {"screen": tmp[:3]}
            if self.cfg.use_labels:
                obs["labels"] = tmp[3:4]
            if self.cfg.use_depth:
                obs["depth"] = tmp[4:5]
            if self.cfg.use_automap:
                obs["automap"] = tmp[5:]
        else:
            obs = {"screen": state.screen_buffer}
            if state.labels_buffer is not None:
                obs["labels"] = state.labels_buffer[None]
            if state.depth_buffer is not None:
                obs["depth"] = state.depth_buffer[None]
            if state.automap_buffer is not None:
                obs["automap"] = state.automap_buffer
        return state, obs, done

    def _update_frame_stack(self, obs, reset: bool = False):
        if reset:
            # clear and populate frame stack
            self.frame_stack.clear()
            for _ in range(self.frame_stack.maxlen):
                self.frame_stack.append(obs)
        else:
            self.frame_stack.append(obs)

        # return stacked observation dictionary
        buffers = {k: [frames[k] for frames in self.frame_stack] for k in obs}
        obs = {k: np.stack(v, 1) for k, v in buffers.items()}  # (c, t, h, w)
        obs = {k: o if o.shape[1] != 1 else o.squeeze(1) for k, o in obs.items()}
        return obs

    def _add_bot(self):
        self.game.send_game_command("removebots")
        for i in range(self.cfg.num_bots):
            self.game.send_game_command("addbot")

    def _update_game_vars(self, reset: bool = False):
        self._game_vars_pre = deepcopy(self._game_vars)
        if self.unwrapped._state is not None:  # ensure current frame is available
            for key in self._game_var_list:
                key_name = str(key)
                self._game_vars[key_name[key_name.find(".") + 1 :]] = (
                    self.game.get_game_variable(key)
                )
            # Fix "HEALTH == -999900.0" error when respawn
            if self._game_vars["HEALTH"] < 0.0:
                self._game_vars["HEALTH"] = 0.0

    def _record_replay(self, obs, reset: bool = False):
        if reset:
            self.replay = defaultdict(list)

        if self.record:
            self.replay["tic"].append(self.game.get_episode_time())
            self.replay["frames"].append(obs["screen"])
            for k in ["labels", "depth", "automap"]:
                if k in obs:
                    self.replay[k].append(obs[k])
            self.replay["game_vars"].append(copy.deepcopy(self._game_vars))


class VizdoomMPEnv(Env):
    """
    Custom VizDoom multiplayer environment for the jku.wad DLR 25 final challenge.

    Parameters:
    - config_path (default: "doom_arena/scenarios/jku.cfg"):
    Path to the VizDoom configuration file. **Do not change** for the challenge.

    - reward_fn (default: None):
    Callable used to compute rewards. To define custom rewards, extend the `__call__`
    method in `doom_arena.reward.VizDoomReward`.

    - num_players (default: 1):
    Number of external agent players. Players connect via sockets and share the same
    game instance, with player one acting as host.

    - num_bots (default: 0):
    Number of NPC enemies in the environment.

    - bot_skill (default: 0):
    Skill level for bots, ranging from 0 to 3 (where 3 represents perfect bots).

    - discrete7 (default: True):
    Enables a simplified action space with only 7 discrete buttons.

    - episode_timeout (default: 10050):
    Maximum number of frames before the episode is terminated.

    - n_stack_frames (default: 1):
    Number of frames to include in the stacked observation buffer.

    - extra_state (default: None):
    Additional sensor observations. Can be one of the following:
        - `ObsBuffer.LABELS`: first-person view with enemies and pickups highlighted.
        - `ObsBuffer.DEPTH`: first-person depth map.
        - `ObsBuffer.AUTOMAP`: top-down view using Doom's automap.

    - doom_map (default: "ROOM"):
    Map to load in the environment. Options:
        - `TRNM`: Medium-sized map used for PvP tournaments.
        - `TRNMBIG`: Large map with multiple interconnected rooms.
        - `ROOM`: Small map for evaluation and grading; supports up to 4 bots.

    - crosshair (default: True):
    Enables the red aiming crosshair.

    - hud (default: "full"):
    Heads-up display level. Options are `"full"`, `"minimal"`, or `"off"` (or `None`).
    """

    def __init__(
        self,
        config_path: str = "doom_arena/scenarios/jku.cfg",
        reward_fn: Optional[Callable] = None,
        num_players: int = 1,
        num_bots: int = 0,
        bot_skill: int = 0,
        discrete7: bool = True,
        episode_timeout: int = 10050,
        n_stack_frames: Union[int, List[int]] = 1,
        extra_state: Optional[Union[List[ObsBuffer], List[List[ObsBuffer]]]] = None,
        doom_map: str = "ROOM",
        crosshair: Sequence[bool] = True,
        hud: Sequence[str] = "full",
        seed: int = 1337,
    ):
        if config_path == "doom_arena/scenarios/jku.cfg":
            assert doom_map in ["TRNM", "TRNMBIG", "ROOM"]
            if doom_map == "ROOM":
                num_bots = min(num_bots, 4)
        else:
            warn(f"Using custom untested configuration : {config_path}.")
        self.num_players = num_players
        self.num_bots = num_bots
        if not isinstance(n_stack_frames, Sequence):
            n_stack_frames = [n_stack_frames] * num_players
        if extra_state is not None and len(extra_state) == 0:
            extra_state = None
        if extra_state is not None and not isinstance(extra_state[0], Sequence):
            extra_state = [extra_state] * num_players
        if not isinstance(crosshair, Sequence):
            crosshair = [crosshair] * num_players
        if not isinstance(hud, List):
            hud = [hud] * num_players
        # select empty port for multiplayer
        self.port = pick_unused_port()
        # host cfg
        self.host_cfg = PlayerHostConfig(self.port)
        self.host_cfg.num_players = num_players
        # join cfg
        self.join_cfg = PlayerJoinConfig(self.port)
        # player cfg
        self.players_cfg = []
        # players
        for i in range(num_players):
            cfg = PlayerConfig()
            cfg.config_path = config_path
            cfg.player_mode = vzd.Mode.PLAYER
            cfg.screen_resolution = vzd.ScreenResolution.RES_256X192
            cfg.screen_format = vzd.ScreenFormat.CRCGCB
            cfg.ticrate = 35
            cfg.crosshair = crosshair[i]
            cfg.respawns = True
            cfg.hud = hud[i]
            cfg.n_stack_frames = n_stack_frames[i]
            cfg.transform = transforms.Compose([to_tensor, resize, minmax, cat_dict])
            cfg.bot_skill = bot_skill
            if extra_state is not None:
                extras = [v.value for v in extra_state[i]]
                cfg.use_labels = ObsBuffer.LABELS.value in extras
                cfg.use_depth = ObsBuffer.DEPTH.value in extras
                cfg.use_automap = ObsBuffer.AUTOMAP.value in extras
            if doom_map is not None:
                cfg.doom_map = doom_map
            cfg.episode_timeout = episode_timeout
            if i == 0:
                cfg.host_cfg = self.host_cfg
                cfg.name = "WhoAmI"
                cfg.num_bots = num_bots
            else:
                cfg.join_cfg = self.join_cfg
                cfg.name = f"P{i}"
            self.players_cfg.append(cfg)

        self.envs = []
        for i, cfg in enumerate(self.players_cfg):
            e = PlayerEnv(cfg, discrete7=discrete7, seed=seed)
            self.envs.append(e)

        if len(self.envs) == 1:
            self.env = self.envs[0]
        else:
            self.env = ParallelEnv(self.envs)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        if reward_fn is None:
            reward_fn = VizDoomReward(num_players)
        self.reward_fn = reward_fn

    def step(self, actions):
        obs, vizdoom_rwds, dones, info = self.env.step(actions)
        if len(self.envs) == 1:
            # single player
            obs = [obs]
            vizdoom_rwds = [vizdoom_rwds]
            dones = [dones]
        done = all(dones)

        rwds = []
        for player_idx in range(self.num_players):
            rwd_p = self.reward_fn(
                vizdoom_rwds[player_idx],
                self.envs[player_idx].unwrapped._game_vars,
                self.envs[player_idx].unwrapped._game_vars_pre,
                player_idx,
            )
            rwds.append(rwd_p)

        # total rewards
        rwds = [sum(rwd_p) for rwd_p in rwds]
        return obs, rwds, done, info

    def reset(self, **kwargs):
        self.reward_fn.reset()
        self.obs = self.env.reset(**kwargs)
        if len(self.envs) == 1:
            self.obs = [self.obs]
        return self.obs

    def enable_replay(self):
        print("Enabling replays!")
        for e in self.envs:
            e.record = True

    def disable_replay(self):
        print("Disabling replays!")
        for e in self.envs:
            e.record = False

    def get_player_replays(self):
        replays = {}
        for e in self.envs:
            if e.record:
                replays[e.cfg.name] = e.replay

        return replays


################################ eval notebook start ###################################


DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Seed random number generators
torch.backends.cudnn.deterministic = True
if os.path.exists("seed.rnd"):
    with open("seed.rnd", "r") as f:
        seed = int(f.readline().strip())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
else:
    seed = 1337


def make_env(seed, config) -> VizdoomMPEnv:
    extra_state = None
    if config["extra_state"] is not None:
        extra_state = []
        for c in config["extra_state"]:
            if c in {e.value for e in ObsBuffer}:
                extra_state.append(ObsBuffer(c))

    env = VizdoomMPEnv(
        num_players=1,
        num_bots=4,
        bot_skill=0,
        doom_map="ROOM",
        episode_timeout=2000,
        n_stack_frames=config["n_stack_frames"],
        extra_state=extra_state,
        hud=config["hud"],
        crosshair=config["crosshair"],
        seed=seed,
    )
    return env


class Agent:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

    @torch.no_grad
    def select_action(self, frames):
        frames = frames.unsqueeze(0).to(DEVICE, dtype=DTYPE)
        logits = self.model(frames)
        if isinstance(logits, tuple):
            logits, _ = logits
        probs = Categorical(logits=logits)
        return probs.sample().cpu().numpy()[0]


def run_episode(agent: Agent, seed: int = 1337):
    env = make_env(seed, config=agent.config)
    obs = env.reset()
    score = 0
    done = False
    while not done:
        obs = obs[0]
        action = agent.select_action(obs)
        obs, rwd, done, _ = env.step(action)
        score += rwd[0]
    env.close()
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    model_file = args.submission
    n_episodes = args.episodes

    # Load model
    model = onnx.load(model_file)
    config = next(
        (json.loads(p.value) for p in model.metadata_props if p.key == "config"), {}
    )
    model = ConvertModel(model)
    model.eval()
    model = model.to(DEVICE, dtype=DTYPE)
    agent = Agent(model, config, DEVICE)

    # Evaluate model
    scores = []
    for i in range(n_episodes):
        if seed is not None:
            seed = np.random.randint(1e7)
        scores.append(run_episode(agent, seed=seed))

    # Print result
    print("\n\n\n")
    print(np.mean(scores))
