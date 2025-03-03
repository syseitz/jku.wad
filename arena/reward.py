from typing import Dict, Tuple

import numpy as np


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
    ) -> Tuple:
        self._step += 1
        _ = game_var, game_var_old, player_id
        rwd_hit = (
            1.0 if (game_var["HITCOUNT"] - game_var_old["HITCOUNT"]) > 0.0 else 0.0
        )
        rwd_frag = 10.0 * (game_var["FRAGCOUNT"] - game_var_old["FRAGCOUNT"])
        return (
            vizdoom_reward,
            rwd_hit,
            rwd_frag,
        )

    def outcome(self):
        # TODO check this out
        leaderboard = []
        for player_id in range(self.num_players):
            leaderboard = np.append(leaderboard, [self.cumulated_frag[player_id]])
        leaderboard_index = np.argsort(-leaderboard)

        outcome = np.zeros(self.num_players)
        if self.num_players == 1:
            # single player
            return outcome
        elif self.num_players == 2:
            # duel
            ind_win = ind_lose = 1
            if self.cumulated_frag[0] == self.cumulated_frag[1]:
                # tie
                return outcome
        else:
            # multiplayer
            ind_win = int(float(self.num_players) * (1 / 3.0))
            ind_lose = int(float(self.num_players) * (2 / 3.0))

        for player_id in range(self.num_players):
            if player_id < ind_win:
                # ranked top, deemed as win
                outcome[leaderboard_index[player_id]] = 1
            elif ind_win <= player_id < ind_lose:
                # ranked middle, deemed as tie
                outcome[leaderboard_index[player_id]] = 0
            else:  # ind_lose <= 1
                # ranked bottom, deemed as lose
                outcome[leaderboard_index[player_id]] = -1
        return outcome


# TODO check if it works
class ArenaVizDoomReward(VizDoomReward):
    """Vizdoom vectorized-reward wrapper, tuned for cig.wad (map 1).

    https://github.com/tencent-ailab/Arena/blob/dev-open/arena/env/vizdoom_env.py#L326
    """

    def reset(self):
        super().reset()
        self.cumulated_navigate = np.zeros(self.num_players)
        self.address_list = np.zeros((self.num_players, 18))
        self.adress_saver = np.zeros(self.num_players)
        self.adress_sum = np.zeros(self.num_players)

    def __call__(
        self,
        vizdoom_reward: float,
        game_var: Dict[str, float],
        game_var_old: Dict[str, float],
        player_id: int,
    ) -> Tuple:
        super().__call__(vizdoom_reward, game_var, game_var_old, player_id)
        _ = vizdoom_reward
        rwd_kill_frag = 0
        rwd_killed_frag = 0
        rwd_hit = 0
        rwd_hitt = 0
        rwd_armo = 0
        rwd_heal = 0
        rwd_shot1 = 0
        rwd_shot2 = 0

        frag_diff = game_var["FRAGCOUNT"] - game_var_old["FRAGCOUNT"]
        if frag_diff > 0:
            rwd_kill_frag = 1
        if frag_diff < 0:
            rwd_killed_frag = -1

        rwd_hit = (
            1.0 if (game_var["HITCOUNT"] - game_var_old["HITCOUNT"]) > 0.0 else 0.0
        )
        hits_taken_diff = game_var["HITS_TAKEN"] - game_var_old["HITS_TAKEN"]
        rwd_hitt = 0.0 if hits_taken_diff == 0.0 else -1.0

        health_diff = game_var["HEALTH"] - game_var_old["HEALTH"]
        if 5 < health_diff <= 25:
            rwd_heal = 1
        ammo_diff = (
            game_var["SELECTED_WEAPON_AMMO"] - game_var_old["SELECTED_WEAPON_AMMO"]
        )

        if ammo_diff == 1 or ammo_diff == 2 or ammo_diff == 5:
            rwd_shot1 = 1
        if ammo_diff == -1 and rwd_hit == 0:
            rwd_shot2 = -1

        armo_diff = game_var["ARMOR"] - game_var_old["ARMOR"]
        if armo_diff > 0.0:
            rwd_armo = 1.0
        move_diff = self.move_distance(
            game_var["POSITION_X"],
            game_var["POSITION_Y"],
            game_var_old["POSITION_X"],
            game_var_old["POSITION_Y"],
        )

        if move_diff > 4.0 and health_diff >= 0:
            rwd_move = 0.001
        elif move_diff < 0.6:
            rwd_move = -0.006
        else:
            rwd_move = -0.0005

        binary_address = self.judge_address(
            game_var["POSITION_X"], game_var["POSITION_Y"], i
        )
        self.adress_sum[player_id] = np.sum(binary_address[player_id], axis=0)

        death_diff = game_var["DEATHCOUNT"] - game_var_old["DEATHCOUNT"]
        if death_diff != 0 or self.adress_sum[player_id] == 18:
            self.address_list[player_id] = np.zeros(18)
            binary_address[player_id] = np.zeros(18)
            self.adress_sum[player_id] = 0
            self.adress_saver[player_id] = 0
        elif self.adress_saver[player_id] != self.adress_sum[player_id]:
            self.adress_saver[player_id] = self.adress_sum[player_id]
            rwd_move = 0
        return (
            rwd_kill_frag,
            rwd_killed_frag,
            rwd_hit,
            rwd_hitt,
            rwd_heal,
            rwd_shot1,
            rwd_shot2,
            rwd_armo,
            rwd_move,
        )

    def outcome(self):
        FRAG_arr = []
        Navigate_arr = []
        for player_id in range(self.num_players):
            FRAG_arr = np.append(FRAG_arr, [self.cumulated_frag[player_id]])
            Navigate_arr = np.append(Navigate_arr, self.cumulated_navigate[player_id])
        FRAG_sort_index = np.argsort(-FRAG_arr)
        outcome = np.zeros(self.num_players)
        assert self.num_players >= 2
        if self.num_players == 2:
            ind_win = ind_lose = 1  # there is no tie
        else:  # >= 3
            ind_win = int(float(self.num_players) * (1 / 3.0))
            ind_lose = int(float(self.num_players) * (2 / 3.0))
        for player_id in range(self.num_players):
            if player_id < ind_win:
                # ranked top, deemed as win
                outcome[FRAG_sort_index[player_id]] = 1
            elif ind_win <= player_id < ind_lose:
                # ranked middle, deemed as tie
                outcome[FRAG_sort_index[player_id]] = 0
            else:  # ind_lose <= i
                # ranked bottom, deemed as lose
                outcome[FRAG_sort_index[player_id]] = -1

        rwds = [(outcome[i],) + rwd_p[1:] for i, rwd_p in enumerate(rwds)]
        return outcome

    def move_distance(self, x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        return np.sqrt(dx * dx + dy * dy)

    def judge_address(self, x: float, y: float, player_id: int):
        # magic
        if -432 < x < -304:
            if 1162 < y < 1296:
                self.address_list[player_id][0] = 1
            elif 784 < y < 885:
                self.address_list[player_id][1] = 1
        elif -304 < x < -16:
            if 400 < y < 600:
                self.address_list[player_id][2] = 1
        elif -16 < x < 80:
            if 1165 < y < 1296:
                self.address_list[player_id][3] = 1
            elif -16 < y < 84:
                self.address_list[player_id][4] = 1
        elif 144 < x < 272:
            if 780 < y < 880:
                self.address_list[player_id][5] = 1
            elif 144 < y < 250:
                self.address_list[player_id][6] = 1
        elif 396 < x < 432 and 1423 < y < 1488:
            self.address_list[player_id][7] = 1
        elif 412 < x < 610:
            if 780 < y < 880:
                self.address_list[player_id][8] = 1
        elif 847 < x < 880 and 1423 < y < 1488:
            self.address_list[player_id][9] = 1
        elif 734 < x < 880:
            if 780 < y < 880:
                self.address_list[player_id][10] = 1
            elif 144 < y < 250:
                self.address_list[player_id][11] = 1
        elif 700 < x < 784 and -336 < y < -230:
            self.address_list[player_id][12] = 1
        elif 944 < x < 1040:
            if 1165 < y < 1296:
                self.address_list[player_id][13] = 1
            elif -16 < y < 84:
                self.address_list[player_id][14] = 1
        elif 1230 < x < 1360:
            if 1165 < y < 1296:
                self.address_list[player_id][15] = 1
            elif 700 < y < 816:
                self.address_list[player_id][16] = 1
            elif 208 < y < 320:
                self.address_list[player_id][17] = 1
        return self.address_list
