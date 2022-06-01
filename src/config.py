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
LOG_FILE = 'final.log'  # set to None for unique file names

# ------------------
# DISPLAY / PLOTTING
# ------------------

ENABLE_PATH_PLOTTING = True  # plot agent path in maze
ENABLE_RESULTS_PLOTTING = True  # plot performance results

# WORLD COLORS:
WALL_COLOR = '#eeeeee'
PATH_COLOR = '#111111'
FIRE_COLOR = 'red'

# AGENT COLORS:
TRAIN_POSITION_HISTORY_COLOR = 'orange'
EVAL_POSITION_HISTORY_COLOR = 'blue'

TRAIN_PERFORMANCE_COLOR = TRAIN_POSITION_HISTORY_COLOR
EVAL_PERFORMANCE_COLOR = EVAL_POSITION_HISTORY_COLOR
MEAN_EVAL_PERFORMANCE_COLOR = EVAL_POSITION_HISTORY_COLOR

# ------------------
#  WORLD PARAMETERS
# ------------------

MAZE_PATH = './mazes/final.npy'  # './mazes/final.npy' is the provided assignment maze # TODO -RENAME


ENABLE_FIRES = True  # enable dynamic fires

ENABLE_FAST_READ_MAZE = True  # enable faster implementation of read_maze algorithm

# world parameters:
HEIGHT = 201
WIDTH = 201
N_ACTIONS = 5

# ------------------
#  AGENT
# ------------------

PRETRAINED_Q_PATH = '../data/pretrained_agent.npy'

USE_OPTIMAL_POLICY = True  # use optimal training policy - (leave this True)

TRAIN_MAX_STEPS = 500_000
EVAL_MAX_STEPS = 100_000

NUM_EPOCHS = 10
NUM_EVAL_EPOCHS = 10

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
