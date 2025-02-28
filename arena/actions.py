import vizdoom as vd

VIZDOOM_BUTTONS = [
    vd.Button.ATTACK,
    vd.Button.TURN_LEFT,
    vd.Button.TURN_RIGHT,
    vd.Button.MOVE_RIGHT,
    vd.Button.MOVE_LEFT,
    vd.Button.MOVE_FORWARD,
    vd.Button.MOVE_BACKWARD,
    vd.Button.SPEED,
    vd.Button.TURN180,
    vd.Button.TURN_LEFT_RIGHT_DELTA,
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
