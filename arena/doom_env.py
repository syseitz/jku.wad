"""
Adapted from https://github.com/tencent-ailab/Arena

Arena: a toolkit for Multi-Agent Reinforcement Learning https://arxiv.org/abs/1907.09467
All credits to authors.
"""

from typing import Callable, Optional, Sequence, List

from copy import deepcopy
from collections import deque
from portpicker import pick_unused_port
import numpy as np
import torch

import vizdoom as vd
import gym
from gym.spaces import Box, MultiDiscrete, Discrete

from arena.parallel import ParallelEnv
from arena.utils import get_action_dim, get_screen_shape
from arena.actions import VIZDOOM_ACTIONS
from arena.reward import VizDoomReward
from arena.player import (
    PlayerHostConfig,
    PlayerJoinConfig,
    PlayerConfig,
    player_setup,
    player_host_setup,
    player_join_setup,
)


class PlayerEnv(gym.Env):
    """ViZDoom per player environment, tuned for CIG Track 1 Death Match.

    Use Wu Yuxing"s trick to enhance the action.
    TODO(pengsun): move the action-enhancement logic to an Interface?"""

    def __init__(
        self, cfg, transforms: Optional[Callable] = None, discrete6: bool = True
    ):
        self.cfg = cfg
        self.game = None
        self.discrete6 = discrete6
        self.transforms = transforms

        # TODO does not account for transforms
        self.observation_space = Box(
            low=0,
            high=255,
            dtype=np.uint8,
            shape=get_screen_shape(cfg.screen_format, cfg.screen_resolution),
        )
        if discrete6:
            # simplified
            self.action_space = Discrete(6)
        else:
            # actual button space
            self.action_space = MultiDiscrete([2] * get_action_dim(cfg))

        self._state = None
        self._obs = None
        self._rwd = None
        self._done = None
        self._act = None
        self._game_var_list = None
        self._game_vars = {}
        self._game_vars_pre = {}

        self.history = deque(maxlen=60)

    def repeat_action(self, action):
        cnt = 1
        while len(self.history) > cnt + 1 and self.history[-(cnt + 1)] == action:
            cnt += 1
        return cnt

    def discrete6_to_button(self, act: int):
        act = int(act)
        return VIZDOOM_ACTIONS[act]

    def adjust_doom_action(self, action):
        """Convert to Discrete(6) action to the full allowed action."""
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
        if not self.cfg.is_multiplayer_game:
            if self.game is None:
                self._init_game()
            self.game.new_episode()
        else:
            self._init_game()
            if self.cfg.num_bots > 0:
                self._add_bot()

        self._state, self._obs, self._done = self._grab()
        self._update_vars()

        # apply (frame) transforms
        if self.transforms is not None:
            self._obs = self.transforms(self._obs)

        return self._obs

    def step(self, action):
        if isinstance(action, np.ndarray) or isinstance(action, torch.Tensor):
            action = action.tolist()
        if self.discrete6:
            action = self.discrete6_to_button(action)
        # vizdoom step
        self._rwd = self.game.make_action(
            self.adjust_doom_action(action), self.cfg.repeat_frame
        )
        self._state, self._obs, self._done = self._grab()
        self._update_vars()

        # apply (frame) transforms
        if self.transforms is not None:
            self._obs = self.transforms(self._obs)

        return self._obs, self._rwd, self._done, {}

    def close(self):
        if self.game:
            self.game.close()

    def render(self, *args):
        return self._obs

    def _init_game(self):
        self.close()

        game = vd.DoomGame()
        game = player_setup(game, self.cfg)
        if self.cfg.is_multiplayer_game:
            if self.cfg.host_cfg is not None:
                game = player_host_setup(game, self.cfg.host_cfg)
            elif self.cfg.join_cfg is not None:
                game = player_join_setup(game, self.cfg.join_cfg)
            else:
                raise ValueError("neither host nor join, error!")
        game.set_window_visible(False)
        game.init()
        self.game = game
        self._game_var_list = self.game.get_available_game_variables()
        self._update_vars()

    def _grab(self):
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        if done:
            obs = np.ndarray(self.observation_space.shape, self.observation_space.dtype)
        else:
            obs = state.screen_buffer
        return state, obs, done

    def _add_bot(self):
        self.game.send_game_command("removebots")
        for i in range(self.cfg.num_bots):
            self.game.send_game_command("addbot")

    def _update_vars(self):
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


class VizdoomMPEnv(gym.Env):
    """ViZDoom multi-player environment."""

    def __init__(
        self,
        config_path: str = "multi.cfg",
        reward_fn: Optional[Callable] = None,
        num_players: int = 2,
        num_bots: int = 0,
        discrete6: bool = True,
        episode_timeout: int = 2000,
        # TODO
        custom_game_variables: Optional[List[str]] = None,
        player_transforms: Optional[Sequence[Callable]] = None,
    ):
        self.num_players = num_players
        self.num_bots = num_bots
        # select empty port for multiplayer
        self.port = pick_unused_port()
        # host cfg
        self.host_cfg = PlayerHostConfig(self.port)
        self.host_cfg.num_players = num_players
        # join cfg
        self.join_cfg = PlayerJoinConfig(self.port)
        # player cfg
        self.players_cfg = []
        for i in range(self.host_cfg.num_players):
            cfg = PlayerConfig()
            cfg.config_path = config_path
            cfg.player_mode = vd.Mode.PLAYER
            cfg.screen_resolution = vd.ScreenResolution.RES_256X192
            cfg.screen_format = vd.ScreenFormat.CBCGCR
            if custom_game_variables is not None:
                cfg.available_game_variables = custom_game_variables
            cfg.episode_timeout = episode_timeout
            if i == 0:  # host
                cfg.host_cfg = self.host_cfg
                cfg.name = "WhoAmI"
                cfg.num_bots = num_bots
            else:
                cfg.join_cfg = self.join_cfg
                cfg.name = "P{}".format(i)
            self.players_cfg.append(cfg)

        if not isinstance(player_transforms, Sequence):
            player_transforms = [player_transforms] * len(self.players_cfg)

        self.envs = []
        for i, cfg in enumerate(self.players_cfg):
            e = PlayerEnv(cfg, transforms=player_transforms[i], discrete6=discrete6)
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
        info["alive"] = dones
        # terminate a single player is alive
        done = sum(dones) >= len(dones)

        rwds = []
        for player_idx in range(self.num_players):
            rwd_p = self.reward_fn(
                vizdoom_rwds[player_idx],
                self.envs[player_idx].unwrapped._game_vars_pre,
                self.envs[player_idx].unwrapped._game_vars,
                player_idx,
            )
            rwds.append(rwd_p)

        if done:
            outcome = self.reward_fn.outcome()
            info["outcome"] = outcome
            rwds = [(outcome[i],) + rwd_p for i, rwd_p in enumerate(rwds)]
        # total rewards
        rwds = [sum(rwd_p) for rwd_p in rwds]
        return obs, rwds, done, info

    def reset(self, **kwargs):
        self.reward_fn.reset()
        self.obs = self.env.reset(**kwargs)
        if len(self.envs) == 1:
            self.obs = [self.obs]
        return self.obs
