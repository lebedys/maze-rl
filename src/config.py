import numpy as np

# ------------------
#   RANDOM SEED
# ------------------

RANDOM_SEED: int = 2022

# ------------------
#   ERROR HANDLING
# ------------------

ENABLE_WARNING_AS_ERROR = False  # catch RuntimeWarning as error
if ENABLE_WARNING_AS_ERROR:
    np.seterr(all='raise')

# ------------------
#    LOGGING
# ------------------

ENABLE_LOGGING = True  # log agent to file

# ------------------
# DISPLAY / PLOTTING
# ------------------

ENABLE_PLOTTING = True  # plot agent path

# WORLD COLORS:
WALL_COLOR = '#eeeeee'
PATH_COLOR = '#111111'
FIRE_COLOR = 'red'

# AGENT COLORS:
TRAIN_POSITION_HISTORY_COLOR = 'orange'
EVAL_POSITION_HISTORY_COLOR = 'blue'

# ------------------
#  WORLD PARAMETERS
# ------------------

MAZE_PATH = './mazes/final.npy'  # './mazes/final.npy' is the provided assignment maze

ENABLE_FIRES = True  # enable dynamic fires

ENABLE_FAST_READ_MAZE = True  # enable faster implementation of read_maze algorithm

# world parameters:
HEIGHT = 201
WIDTH = 201
N_ACTIONS = 5

# ------------------
#  AGENT
# ------------------

PRETRAINED_Q_PATH = '../data/agents/q_pretrained.npy'
# todo - make absolute out of above

USE_OPTIMAL_POLICY = False  # use optimal training policy

TRAIN_MAX_STEPS = 500_000
EVAL_MAX_STEPS = 100_000

REWARDS = {  # agent rewards

    # WIN CONDITION:
    'finish': 10.,  # finish maze

    # obstacles:
    'wall': -np.inf,  # hit wall
    'fire': -0.0,  # hit fire

    # movement:
    'step_taken': -0.,  # take step in any direction
    'stay': -0.,  # penalty for staying in place

    # backtracking:
    'repeat_step': -0.5,  # backtracking penalty
    'revisited': -0.0,  # revisit previously-visited node
    'unvisited': 0.,  # visit unvisited node

    # distance:
    'distance_to_end': -0.,  # penalize distance to end position
    'distance_from_start': 0.  # reward distance from initial position
}
