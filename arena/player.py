import vizdoom as vd


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
        self.is_render_hud = None
        self.screen_resolution = None
        self.screen_format = None
        self.is_window_visible = None
        self.ticrate = None
        self.episode_timeout = None
        self.name = None
        self.colorset = None
        self.available_game_variables = [
            vd.GameVariable.HEALTH,
            vd.GameVariable.AMMO3,
            # additional
            vd.GameVariable.FRAGCOUNT,
            vd.GameVariable.ARMOR,
            vd.GameVariable.HITCOUNT,
            vd.GameVariable.HITS_TAKEN,
            vd.GameVariable.DEAD,
            vd.GameVariable.DEATHCOUNT,
            vd.GameVariable.DAMAGECOUNT,
            vd.GameVariable.DAMAGE_TAKEN,
            vd.GameVariable.KILLCOUNT,
            vd.GameVariable.SELECTED_WEAPON,
            vd.GameVariable.SELECTED_WEAPON_AMMO,
            vd.GameVariable.POSITION_X,
            vd.GameVariable.POSITION_Y,
        ]

        self.repeat_frame = 2
        self.num_bots = 0

        self.is_multiplayer_game = True
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
    if player_config.is_window_visible is not None:
        game.set_window_visible(player_config.is_window_visible)
    if player_config.ticrate is not None:
        game.set_ticrate(player_config.ticrate)
    if player_config.episode_timeout is not None:
        game.set_episode_timeout(player_config.episode_timeout)
    if player_config.episode_timeout is not None:
        game.set_episode_timeout(player_config.episode_timeout)

    # custom game variables
    if player_config.available_game_variables is not None:
        game.set_available_game_variables(player_config.available_game_variables)

    game.set_console_enabled(False)

    if player_config.name is not None:
        game.add_game_args("+name {}".format(player_config.name))
    if player_config.colorset is not None:
        game.add_game_args("+colorset {}".format(player_config.colorset))
    return game
