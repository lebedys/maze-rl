import src.lib.read_maze as rm
import src.lib.fast_read_maze as frm

import numpy as np
import random

import time

from src.config import RANDOM_SEED
# NUMPY_SEED = 1234

# from src.lib.read_maze import maze_cells

def test_rm(maze_cells: np.ndarray,
            iterations: int = 100_000):  # read maze
    rm_observations = []
    rm_positions = []

    for i in range(1, iterations):
        row, col = np.random.randint(1, 199, size=2)
        rm_positions.append(np.array([row, col]))
        new_maze_cells, observation = rm.get_local_maze_information(maze_cells, row, col)
        rm_observations.append(observation)

    return new_maze_cells, rm_positions, rm_observations

def test_frm(maze_cells: np.ndarray,
             iterations: int = 100_000):  # read maze
    frm_observations = []
    frm_positions = []

    for i in range(1, iterations):
        row, col = np.random.randint(1, 199, size=2)
        frm_positions.append(np.array([row, col]))
        new_maze_cells, observation = frm.get_local_maze_information(maze_cells, row, col)
        frm_observations.append(observation)

    return new_maze_cells, frm_positions, frm_observations


if __name__ == '__main__':

    iterations = 10_000

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    maze_cells = rm.load_maze('../mazes/final.npy')

    rm_t1 = time.process_time_ns()
    maze_cells, rm_positions, rm_observations = test_rm(maze_cells, iterations)
    rm_t2 = time.process_time_ns()
    rm_elapsed = rm_t2 - rm_t1
    print('elapsed (RM):', rm_elapsed, 'ns')

    rm_maze_cells = maze_cells.copy()

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    maze_cells = frm.load_maze('../mazes/final.npy')

    frm_t1 = time.process_time_ns()
    maze_cells, frm_positions, frm_observations = test_frm(maze_cells, iterations)
    frm_t2 = time.process_time_ns()
    frm_elapsed = frm_t2 - frm_t1
    print('elapsed (fast RM):', frm_elapsed, 'ns')

    frm_maze_cells = maze_cells.copy()

    rm_positions = np.array(rm_positions)
    rm_observations = np.array(rm_observations)

    frm_positions = np.array(frm_positions)
    frm_observations = np.array(frm_observations)

    comparison = (rm_positions == frm_positions)
    are_equal = comparison.all()
    print('positions are equal: ', are_equal)

    comparison = (rm_observations == frm_observations)
    are_equal = comparison.all()
    print('observations are equal: ', are_equal)

    saved_time = rm_elapsed - frm_elapsed
    print('saved: {} seconds'.format(saved_time / 1_000_000_000))
    print('faster / slower = {:.2f}%'.format((frm_elapsed/rm_elapsed) * 100))

