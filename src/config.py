ENABLE_LOGGING = True  # log agent to file
ENABLE_PLOTTING = True  # plot agent path
ENABLE_FIRES = False  # enable dynamic fires

PRETRAINED_Q_PATH = '../data/agents/q_pretrained.npy'

NUMPY_SEED: int = 2022

MAZE_PATH = './mazes/final.npy'
ENABLE_FAST_READ_MAZE = True  # enable faster implementation of read_maze algorithm

TRAIN_MAX_STEPS = 1_000_000

# world parameters:
HEIGHT = 201
WIDTH = 201
N_ACTIONS = 5

# default agent rewards:
REWARDS = {
    'finish': 10.,  # finish maze (win condition)

    # todo - penalize reaching max step_count

    # obstacles:
    'wall': -1.0,  # hit wall
    'fire': -1.0,  # hit fire

    # movement:
    'step_taken': -0.,  # take step in any direction
    'stay': -0.,  # stay in place

    # backtracking:
    'repeat_step': -0.5,  # reverse last action # todo - needs better name
    'revisited': -0.05,  # revisit previously-visited node
    'unvisited': 0.,  # todo - reward unvisited

    # distance metrics:
    'distance_to_end': -0.,
    'distance_from_start': 0.
    # todo - change from euclidian to manhattan distance?
}

# DISPLAY PARAMETRS:
WALL_COLOR = '#eeeeee'
PATH_COLOR = '#111111'
FIRE_COLOR = 'red'
# AGENT history:
TRAIN_POSITION_HISTORY_COLOR = 'orange'
EVAL_POSITION_HISTORY_COLOR = 'blue'