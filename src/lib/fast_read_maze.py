import os
import numpy as np
import random

# import src.lib.read_maze as rm

from enum import Enum

from src.config import NUMPY_SEED

random.seed(NUMPY_SEED)  # set seed from config file
np.random.seed(NUMPY_SEED)

flag_list_mapping = {  # currently unused (here for future optimization)
    0: (0, 0),
    1: (0, 1),
    2: (0, 2),
    3: (1, 0),
    5: (1, 2),
    6: (2, 0),
    7: (2, 1),
    8: (2, 2),
}

flag_list = np.array([0, 1, 2, 3, 5, 6, 7, 8])
#  [0, 1, 2
#   3,    5
#   6, 7, 8]

time_list = np.array([0, 1, 2])


class PathType(Enum):
    WALL = 0.0,
    PATH = 1.0


# maze_cells = np.zeros((201, 201, 2), dtype=int)


# load maze
def load_maze(maze_file_path='mazes/final.npy'):
    # todo - this can be optimized

    if not os.path.exists(maze_file_path):
        raise ValueError("Cannot find %s" % maze_file_path)

    else:
        # global maze_cells
        maze = np.load(maze_file_path, allow_pickle=False, fix_imports=True)
        maze_cells = np.zeros((maze.shape[0], maze.shape[1], 2), dtype=int)
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                maze_cells[i][j][0] = maze[i][j]
                # load the maze, with 1 denoting an empty location and 0 denoting a wall
                maze_cells[i][j][1] = 0
                # initialized to 0 denoting no fire

    return maze_cells

def map_flag_to_location(flag: int):  # return
    return flag_list_mapping[flag]


def get_local_maze_information(maze_cells: np.ndarray,
                               row: int, col: int) -> np.ndarray:
    random_flag = random.choice(flag_list)

    # --- DECREMENT FIRE (OPTIMIZED):

    # create mask of cells where fires are present:
    fire_mask = (maze_cells[:, :, 1] > 0)
    # fire_locs = rm.maze_cells[:, :, 1].nonzero()
    # if fire_locs[0].any():
    #     print(fire_locs)
    maze_cells[:, :, 1][fire_mask] -= 1  # decrement masked cells

    # get segment of cells at (row, col) position in maze:
    around = maze_cells[row - 1: (row + 1) + 1, col - 1: (col + 1) + 1, :].astype(int)

    # --- GENERATE NEW FIRE (UN-OPTIMIZED):

    for i in range(3):
        for j in range(3):
            if i == random_flag // 3 and j == random_flag % 3:
                if around[i][j][0] == 0:  # this cell is a wall
                    continue
                ran_time = random.choice(time_list)
                around[i][j][1] = ran_time + around[i][j][1]
                maze_cells[row - 1 + i][col - 1 + j][1] = around[i][j][1]

    return maze_cells, around

# todo - unfinished previous optimization code:
# spawn new fire:
# new_fire_row, new_fire_col = map_flag_to_location(random_flag)
#
# if around[new_fire_row, new_fire_col, 0] != PathType.WALL:
#     random_fire_time = random.choice(time_list)
#     around[new_fire_row, new_fire_col, 1] += random_fire_time
#
# rm.maze_cells[(row - 1):(row + 1) + 1, (col - 1):(col + 1) + 1, 1] = around[:, :, 1]
