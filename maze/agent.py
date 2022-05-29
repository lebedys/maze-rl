import random

import numpy as np
import torch
from typing import Tuple
from collections import defaultdict

from enum import Enum

from maze.lib.util import is_fire, is_wall, euclidian_cost


# todo - implement periodic logging

# directionality:
# [0, 1, 2
#  3, 4, 5
#  6, 7, 8]

class Direction(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

    NONE = 0


directions = np.array(
    [Direction.NONE, Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
)

# todo - move to config.py
# world parameters:
HEIGHT = 11
WIDTH = 11
N_ACTIONS = 5

# default agent rewards:
REWARDS = {
    'finish': 1.,  # finish maze (win condition)

    # obstacles:
    'wall': -1.0,  # hit wall
    'fire': -1.0,  # hit fire

    # movement:
    'step_taken': -0.,  # take step in any direction
    'stay': -1.,  # stay in place

    # backtracking:
    'repeat_step': -0.5,  # reverse last action # todo - needs better name
    'revisited': -0.1,  # revisit previously-visited node
    'unvisited': 0.,  # todo - reward unvisited

    # distance metrics:
    'distance_to_end': -0.,
    'distance_from_start': 0.
    # todo - change from euclidian to manhattan distance?
}


class Agent:

    def __init__(self,
                 start_position: Tuple[int, int] = (1, 1),  # initial position
                 end_position: Tuple[int, int] = (199, 199),  # final position
                 max_steps: int = 10_000_000,  # maximum path length
                 Q: np.ndarray = None,  # Q-table
                 rewards: dict = REWARDS,  # rewards
                 discount: float = 0.5,  # discount factor
                 learning_rate: float = 0.1,  # q-table learning rate
                 ):

        self.start_position = np.array(list(start_position), dtype=int)
        self.end_position = np.array(list(end_position), dtype=int)
        self.Q = Q
        self.rewards = rewards
        self.learning_rate = learning_rate
        self.discount = discount  # discount factor (gamma)

        # initialize agent position:
        self.position = np.copy(self.start_position)  # initial position

        # logging:
        self.step_count = 0
        self.wall_hits = 0

        # todo - convert from tuple-indexed dict, to sequential row-col indexing
        self.visited = defaultdict(int)  # dictionary of visited nodes
        self.previous_position = np.array([0, 0])

        if self.Q is None:  # if no predefined Q-table provided
            # self.Q = np.full((WIDTH, HEIGHT, N_ACTIONS), 1. / N_ACTIONS)  # initialize default q values equal for all
            self.Q = 0.5 * np.random.normal(loc=0.5, scale=0.1, size=(WIDTH, HEIGHT, N_ACTIONS))
            # self.Q[1, 1, :] = np.random.rand(5)  # assign random q-values for start position
            self.Q[1, 1, :] = np.array([
                0, 0, np.random.rand(), np.random.rand(), 0
            ])
            # todo - add noise or some randomness

    def reset_position(self) -> None:
        self.position = np.copy(self.start_position)  # initial position
        self.step_count = 0
        self.visited = defaultdict(int)

    def is_finished(self) -> bool:
        is_finished = (self.position == self.end_position).all()
        return is_finished

    def observe(self, maze: np.ndarray) -> np.ndarray:
        # [[0, 1, 2],
        #  [3, 4, 5],
        #  [6, 7, 8]]

        row, col = self.position
        return maze[row - 1: (row + 1) + 1, col - 1: (col + 1) + 1]

    def invalidate_walls(self, q_values, walls, fires):

        walls = np.array([0., walls[0, 1], walls[1, 2], walls[2, 1], walls[1, 0]])

        q_values[walls == 0.0] = -np.inf

        # todo - deal with fires

        return q_values

    def step(self,
             walls: np.ndarray,  # local observation of walls
             fires: np.ndarray,  # local observation of fires
             train=False,  # enable training
             epsilon: float = 0.0,  # todo - exploration
             q_noise: float = 0.0,  # todo - random noise
             ) -> bool:  # random exploration probability

        prev_row, prev_col = self.previous_position[0], self.previous_position[1]  # decompose previous position

        # get Q(s) vector:
        current_row, current_col = self.position
        q_values = self.Q[current_row, current_col, :]

        # consider only valid directions:
        q_values = self.invalidate_walls(q_values, walls, fires)

        if np.random.random() < epsilon:  # random exploration
            chosen_q_index = np.random.randint(0, 5)
            random_choice = True
            print('random choice')
        else:
            # todo - random noise applied to q value vector
            # pick maximum q-value from Q(s):
            chosen_q_index = np.argmax(q_values)  # select index of highest q-value
            random_choice = False

        chosen_q = q_values[chosen_q_index]  # corresponding q value
        chosen_direction = directions[chosen_q_index]  # choose corresponding direction

        reward = self.rewards['step_taken']  # add initial penalty for making any action
        # todo - should this be applied even when not moving?

        next_row, next_col = current_row, current_col
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
        if is_wall(next_cell):  # penalize wall hit
            self.wall_hits += 1
            print('hit wall at {},{}'.format(current_row, current_col))
            reward += self.rewards['wall']
            next_row, next_col = current_row, current_col  # revert position
        elif is_fire(next_cell):  # penalize fire hit
            reward += self.rewards['fire']
            next_row, next_col = current_row, current_col  # revert position

        if self.step_count is not 0:  # ignore initial step
            if next_row == prev_row and next_col == prev_col:
                reward += self.rewards['repeat_step']  # penalize for repeating most recent step

        if (next_row, next_col) in self.visited:
            # todo - proportional to number of visitations?
            reward += self.rewards['revisited'] * self.visited[(next_row, next_col)]

        distance_from_start = euclidian_cost(self.position, self.start_position)
        reward += self.rewards['distance_from_start'] * distance_from_start
        distance_to_end = euclidian_cost(self.position, self.end_position)
        reward += self.rewards['distance_to_end'] * distance_to_end

        is_finished = False
        if self.is_finished():
            reward += self.rewards['finish']
            is_finished = True

        if train:  # update q-table
            max_q = self.Q[next_row, next_col, :].max()
            current_q_value = self.Q[current_row, current_col, chosen_q_index]

            next_q_value = current_q_value + self.learning_rate * (reward - chosen_q + self.discount * max_q)
            # next_q_value = max(0, next_q_value)  # set minimum to 0

            self.Q[current_row, current_col, chosen_q_index] = next_q_value

        self.position = np.array([next_row, next_col])  # update position
        self.step_count += 1

        self.visited[(current_row, current_col)] += 1  # increment visited nodes

        self.previous_position = np.copy(self.position)  # update memory of previous position

        return is_finished

    def train(self,
              maze: np.ndarray,
              max_steps: int = 10  # maximum training steps per epoch
              ) -> np.ndarray:

        self.reset_position()  # initialize agent

        # initialize training path:
        train_path = np.empty(shape=(max_steps + 1, 2), dtype=int)  # empty path
        train_path[0, :] = np.copy(self.position)  # add initial position

        for i in range(max_steps):
            observation = self.observe(maze)  # get surrounding information
            walls = observation
            fires = np.array([])

            is_finished = self.step(walls=walls, fires=fires, train=True)  # update position
            train_path[i + 1, :] = np.copy(self.position)  # store new position

            if is_finished:  # achieved goal?
                train_path = train_path[:i + 2, :]  # truncate if path unfilled before returning
                print('finished training in {} steps!'.format(self.step_count))
                break  # end training

        print('training loop broken after {} steps.'.format(self.step_count))

        return train_path  # return path when finished

    def run(self, maze, max_steps=10):  # run pretrained agent
        self.reset_position()  # initialize agent

        run_path = np.empty(shape=(max_steps + 1, 2), dtype=int)
        run_path[0, :] = self.position  # add initial position

        for i in range(max_steps):
            observation = self.observe(maze)  # get surrounding information
            walls = observation
            fires = np.array([])

            is_finished = self.step(walls=walls, fires=fires, train=False)  # update position
            run_path[i + 1, :] = np.copy(self.position)  # store new position

            if is_finished:  # achieved goal?
                run_path = run_path[:i + 2, :]  # truncate if path unfilled before returning
                print('finished running in {} steps!'.format(self.step_count))
                break  # end training

        print('run loop broken after {} steps.'.format(self.step_count))

        return run_path  # return path when finished
