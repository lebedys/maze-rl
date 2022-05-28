import random

import numpy as np
import torch
from typing import Tuple

from enum import Enum

from maze.lib.util import is_fire, is_wall, euclidian_cost


# directionality:
# [0, 1, 2
#  3, 4, 5
#  6, 7, 8]

class Direction(Enum):
    UP = 1
    RIGHT = 5
    DOWN = 7
    LEFT = 3

    NONE = 4


directions = np.array(
    [Direction.NONE, Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
)

# todo - move to config.py
# world parameters:
HEIGHT = 201
WIDTH = 201
N_ACTIONS = 5

# default agent rewards:
REWARDS = {
    'finish': 1.,  # finishing maze

    # obstacles:
    'wall': -0.1,  # hit wall
    'fire': -1.,  # hit fire

    # movement:
    'step_taken': -0.,  # take step in any direction
    'stay': -0.,  # staying in place
    'return': -0.,  # take step backwards
    'visited': 0,  # todo - penalize return to visited position
    # todo - penalize path length?
}


class Agent:

    def __init__(self,
                 start_position: Tuple[int, int] = (1, 1),  # initial position
                 end_position: Tuple[int, int] = (199, 199),  # final position
                 max_steps: int = 10_000_000,   # maximum path length
                 Q: np.ndarray = None,          # Q-table
                 rewards: dict = REWARDS,       # rewards
                 discount: float = 0.1,         # discount factor
                 learning_rate: float = 1,
                 euclidian_cost_weighting: float = 0.0,  # todo - explore impact of this parameter
                 ):

        self.start_position = np.array(list(start_position), dtype=int)
        self.end_position = np.array(list(end_position), dtype=int)
        self.Q = Q
        self.rewards = rewards
        self.learning_rate = learning_rate
        self.discount = discount  # discount factor (gamma)

        self.euclidian_cost_weighting = euclidian_cost_weighting

        # initialize agent position:
        self.position = np.copy(self.start_position)  # initial position
        self.step_count = 0

        if self.Q is None:  # if no predefined Q-table provided
            self.Q = np.full((WIDTH, HEIGHT, N_ACTIONS), 1. / N_ACTIONS)  # initialize default q values equal for all
            self.Q[1, 1, :] = np.random.rand(5)  # assign random q-values for start position
            # todo - add noise or some randomness

    def reset_position(self) -> None:
        self.position = np.copy(self.start_position) # initial position

    def is_finished(self) -> bool:
        is_finished = (self.position == self.end_position).all()
        if is_finished: print('finished!')
        return is_finished

    def observe(self, maze: np.ndarray) -> np.ndarray:
        # [[0, 1, 2],
        #  [3, 4, 5],
        #  [6, 7, 8]]

        row, col = self.position
        return maze[row - 1: (row + 1) + 1, col - 1: (col + 1) + 1]

    def step(self,
             walls: np.ndarray,  # local observation of walls
             fires: np.ndarray,  # local observation of fires
             train=False,        # enable training
             epsilon: float = 0.1,  # todo - exploration
             q_noise: float = 0.0,  # todo - random noise
             ) -> None:  # random exploration probability

        # todo - epsilon for random choice selection
        # todo - add random noise to q values
        # todo - invalidate some directions

        # get Q(s) vector:
        row, col = self.position
        q_values = self.Q[row, col, :]

        # if np.random.random() < epsilon:  # random exploration
        if False:
            chosen_q_index = np.random.randint(0, 5)
            # fixme - need q index chosen
        else:
            # todo - random noise applied to q value vector
            # pick maximum q-value from Q(s):
            chosen_q_index = np.argmax(q_values)  # select index of highest q-value

        chosen_q = q_values[chosen_q_index]  # corresponding q value
        chosen_direction = directions[chosen_q_index]  # choose corresponding direction

        reward = self.rewards['step_taken']  # add initial penalty for making any action
        # todo - should this be applied even when not moving?

        # flatten observation: [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # walls, fires = observation
        # walls = walls.flatten()
        # fires = fires.flatten()

        next_row, next_col = row, col
        if chosen_direction == Direction.NONE:  # no change in position
            reward += self.rewards['stay']  # penalty for staying in place
            next_cell = walls[1, 1]
        elif chosen_direction == Direction.UP:
            next_row -= 1  # move up
            next_cell = walls[0, 1]
        elif chosen_direction == Direction.RIGHT:
            next_col += 1  # move right
            next_cell = walls[1, 2]
        elif chosen_direction == Direction.DOWN:
            next_row += 1  # move down
            next_cell = walls[2, 1]
        elif chosen_direction == Direction.LEFT:
            next_col -= 1  # move left
            next_cell = walls[1, 0]
        else:
            raise ValueError('Directionality value not valid.')

        # penalize hitting obstacles:
        # next_cell = walls[next_row-row, next_col-col]
        if is_wall(next_cell):    # penalize wall hit
            reward += self.rewards['wall']
            next_row, next_col = row, col  # revert position
        elif is_fire(next_cell):  # penalize fire hit
            reward += self.rewards['fire']
            next_row, next_col = row, col  # revert position

        # previous_position = 0
        # if (next_position == previous_position).all():
        #     pass  # todo - check and penalize if returning to previous move

        # todo - small negative reward for distance to goal
        # reward += euclidian_cost(self.position, self.end_position) * self.euclidian_cost_weighting

        # estimate of maximum future q-value:
        max_q = self.Q[next_row, next_col, :].max()

        self.Q[row, col, chosen_q_index] += \
            self.learning_rate * (reward - chosen_q + self.discount * max_q)

        self.position = np.array([next_row, next_col])  # update position

    def train(self,
              maze: np.ndarray,
              max_steps: int = 10  # maximum training steps per epoch
              ) -> np.ndarray:

        self.reset_position()  # initialize agent

        # initialize training path:
        train_path = np.empty(shape=(max_steps+1, 2), dtype=int)  # empty path
        train_path[0, :] = np.copy(self.position)  # add initial position

        for i in range(max_steps):
            observation = self.observe(maze)  # get surrounding information
            walls = observation
            fires = np.array([])

            self.step(walls=walls, fires=fires, train=True)  # update position
            train_path[i+1, :] = np.copy(self.position)  # store new position

            if self.is_finished():  # achieved goal?
                train_path = train_path[:i+2, :]  # truncate if path unfilled before returning
                break  # end training

        return train_path  # return path when finished

    def run(self, maze, max_steps=10):  # run pretrained agent
        self.reset_position()  # initialize agent

        run_path = np.empty(shape=(max_steps+1, 2), dtype=int)
        run_path[0, :] = self.position  # add initial position

        for i in range(max_steps):
            self.step(maze, train=False)

            run_path[i, :] = self.position  # append position

            if self.is_finished():  # achieved goal?
                run_path = run_path[:i+1, :]  # truncate path if unfilled before returning
                break  # end run

        return run_path  # return path when finished