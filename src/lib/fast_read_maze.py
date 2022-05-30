import os
import numpy as np
import random

import src.lib.read_maze as rm

from enum import Enum

from src.config import NUMPY_SEED

random.seed(NUMPY_SEED)  # set seed from config file

flag_list_mapping = {
    0: (0, 0),
    1: (0, 1),
    2: (0, 2),
    3: (1, 0),
    5: (1, 2),
    6: (2, 0),
    7: (2, 1),
    8: (2, 2),
}

flag_list = [0, 1, 2, 3, 5, 6, 7, 8]
#  [0, 1, 2
#   3,    5
#   6, 7, 8]

time_list = [0, 1, 2]


class PathType(Enum):
    WALL = 0.0,
    PATH = 1.0


def map_flag_to_location(flag: int):  # return
    return flag_list_mapping[flag]


def get_local_maze_information(row: int, col: int) -> np.ndarray:
    random_flag = random.choice(flag_list)

    # OPTIMIZED FIRE DECREMENTING:
    # create mask of cells where fires are present:
    fire_mask = (rm.maze_cells[:, :, 1] > 0)
    fire_locs = rm.maze_cells[:, :, 1].nonzero()
    # if fire_locs[0].any():
    #     print(fire_locs)
    rm.maze_cells[:, :, 1][fire_mask] -= 1  # decrement masked cells

    # get segment of cells at (row, col) position in maze:
    around = rm.maze_cells[row - 1: (row + 1) + 1, col - 1: (col + 1) + 1, :].astype(int)

    # spawn new fire:
    # new_fire_row, new_fire_col = map_flag_to_location(random_flag)
    #
    # if around[new_fire_row, new_fire_col, 0] != PathType.WALL:
    #     random_fire_time = random.choice(time_list)
    #     around[new_fire_row, new_fire_col, 1] += random_fire_time
    #
    # rm.maze_cells[(row - 1):(row + 1) + 1, (col - 1):(col + 1) + 1, 1] = around[:, :, 1]

    # todo - optimize further:
    # around = np.zeros((3, 3, 2), dtype=int)
    x, y = row, col
    random_location = random_flag
    for i in range(3):
        for j in range(3):
            # if x - 1 + i < 0 or x - 1 + i >= rm.maze_cells.shape[0] or y - 1 + j < 0 or y - 1 + j >= rm.maze_cells.shape[1]:
            #     around[i][j][0] = 0  # this cell is outside the maze, and we set it to a wall
            #     around[i][j][1] = 0
            #     continue
            # around[i][j][0] = rm.maze_cells[x - 1 + i][y - 1 + j][0]
            # around[i][j][1] = rm.maze_cells[x - 1 + i][y - 1 + j][1]
            if i == random_location // 3 and j == random_location % 3:
                if around[i][j][0] == 0:  # this cell is a wall
                    continue
                ran_time = random.choice(time_list)
                around[i][j][1] = ran_time + around[i][j][1]
                rm.maze_cells[x - 1 + i][y - 1 + j][1] = around[i][j][1]

    return around
