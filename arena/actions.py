import vizdoom as vzd

VIZDOOM_BUTTONS = [
    vzd.Button.ATTACK,
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.SPEED,
    vzd.Button.TURN180,
    vzd.Button.TURN_LEFT_RIGHT_DELTA,
]

# first 6 are core actions
# TODO add more eg change weapon
VIZDOOM_ACTIONS = [
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],  # 0 move fast forward
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 fire
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # 2 move left
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # 3 move right
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # 4 turn left
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # 5 turn right
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 20],  # 6 turn left 40 degree and move forward
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 20],  # 7 turn right 40 degree and move forward
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 8 move forward
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 9 turn 180
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 10 move left
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 11 move right
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 12 turn left
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 13 turn right
]
