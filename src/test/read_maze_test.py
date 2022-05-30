import src.lib.read_maze as rm
import src.lib.fast_read_maze as frm

import numpy as np
import random

import time

from src.config import NUMPY_SEED

# from src.lib.read_maze import maze_cells

def test_rm(iterations: int = 100_000):  # read maze
    rm_observations = []
    rm_positions = []

    for i in range(1, iterations):
        row, col = np.random.randint(1, 199, size=2)
        rm_positions.append(np.array([row, col]))
        observation = rm.get_local_maze_information(row, col)
        rm_observations.append(observation)

    return rm_positions, rm_observations

def test_frm(iterations: int = 100_000):  # read maze
    frm_observations = []
    frm_positions = []

    for i in range(1, iterations):
        row, col = np.random.randint(1, 199, size=2)
        frm_positions.append(np.array([row, col]))
        observation = frm.get_local_maze_information(row, col)
        frm_observations.append(observation)

    return frm_positions, frm_observations


if __name__ == '__main__':

    iterations = 100_000

    np.random.seed(NUMPY_SEED)
    random.seed(NUMPY_SEED)

    rm.load_maze('../mazes/final.npy')

    rm_t1 = time.process_time_ns()
    rm_positions, rm_observations = test_rm(iterations)
    rm_t2 = time.process_time_ns()
    rm_elapsed = rm_t2 - rm_t1
    print('elapsed (RM):', rm_elapsed)

    rm_maze_cells = rm.maze_cells.copy()

    np.random.seed(NUMPY_SEED)
    random.seed(NUMPY_SEED)

    rm.load_maze('../mazes/final.npy')
    frm_t1 = time.process_time_ns()
    frm_positions, frm_observations = test_frm(iterations)
    frm_t2 = time.process_time_ns()
    frm_elapsed = frm_t2 - frm_t1
    print('elapsed (fast RM):', frm_elapsed)

    frm_maze_cells = rm.maze_cells.copy()

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