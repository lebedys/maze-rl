import random

import numpy as np
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
HEIGHT = 201
WIDTH = 201
N_ACTIONS = 5

# default agent rewards:
REWARDS = {
    'finish': 1.,  # finish maze (win condition)

    # todo - penalize reaching max step_count

    # obstacles:
    'wall': -1.0,  # hit wall
    'fire': -1.0,  # hit fire

    # movement:
    'step_taken': -0.1,  # take step in any direction
    'stay': -0.,  # stay in place

    # backtracking:
    'repeat_step': -0.2,  # reverse last action # todo - needs better name
    'revisited': -0.2,  # revisit previously-visited node
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

                 # parameters:
                 Q: np.ndarray = None,  # Q-table

                 # rewards:
                 rewards: dict = REWARDS,  # rewards

                 # hyper-parameters:
                 learning_rate: float = 0.1,  # q-table learning rate
                 discount: float = 0.5,  # discount factor
                 exploration_epsilon: float = 0.1  # probability of random exploration
                 ):

        self.start_position = np.array(list(start_position), dtype=int)
        self.end_position = np.array(list(end_position), dtype=int)
        self.Q = Q
        self.rewards = rewards

        self.learning_rate = learning_rate

        self.discount = discount  # discount factor (gamma)

        self.exploration_epsilon = exploration_epsilon  # probability of random exploration

        # initialize agent position:
        self.position: np.ndarray = np.copy(self.start_position)  # initial position

        # logging parameters:
        self.step_count: int = 0
        self.wall_hit_count: int = 0
        self.fire_hit_count: int = 0

        # dictionary of visited nodes:
        self.visited = defaultdict(int)

        # memory of previous step:
        self.previous_position = np.array([0, 0])
        self.previous_q_index = Direction.NONE

        if self.Q is None:  # if no predefined Q-table provided
            self.Q = np.full((WIDTH, HEIGHT, N_ACTIONS), 0.)

            # experimented with random normal initial values:
            # self.Q = 0.5 * np.random.normal(loc=0.5, scale=0.1, size=(WIDTH, HEIGHT, N_ACTIONS))

            # special values for initial position in q-table:
            self.Q[1, 1, :] = np.array([
                0,  # NONE
                0,  # UP
                np.random.normal(loc=0.5, scale=0.1),  # RIGHT
                np.random.normal(loc=0.5, scale=0.1),  # DOWN
                0,  # LEFT
            ])

    def reset_position(self) -> None:
        self.position = np.copy(self.start_position)  # initial position
        self.step_count = 0
        self.visited = defaultdict(int)

    def is_finished(self) -> bool:
        return (self.position == self.end_position).all()

    def observe(self, maze: np.ndarray) -> np.ndarray:
        # [[0, 1, 2],
        #  [3, 4, 5],
        #  [6, 7, 8]]

        row, col = self.position
        return maze[row - 1: (row + 1) + 1, col - 1: (col + 1) + 1]

        # todo - implement fires

    def invalidate_walls(self, q_values: np.ndarray, walls: np.ndarray, fires: np.ndarray) -> np.ndarray:

        walls = np.array([0.0,  # invalidate Direction.NONE
                          walls[0, 1],  # invalidate all walls
                          walls[1, 2],
                          walls[2, 1],
                          walls[1, 0]
                          ])

        q_values[walls == 0.0] = -np.inf

        # todo - invalidate fires

        return q_values

    def step(self,
             walls: np.ndarray,  # local observation of walls
             fires: np.ndarray,  # local observation of fires
             train=False,  # enable training (updating Q-table)
             ) -> bool:  # random exploration probability

        reward = self.rewards['step_taken']  # add initial penalty for making any action

        prev_row, prev_col = self.previous_position  # decompose previous position
        current_row, current_col = self.position  # decompose current position

        is_finished = False
        if self.is_finished():
            reward += self.rewards['finish']
            is_finished = True

            next_row, next_col = self.position
        else:

            # select Q(s) vector:
            q_values = self.Q[current_row, current_col, :]

            # consider only valid directions:
            q_values = self.invalidate_walls(q_values, walls, fires)
            # todo - invalidate fires

            random_choice = False
            if np.random.random() < self.exploration_epsilon:  # random exploration
                # todo - only consider valid actions here
                chosen_q_index = np.random.randint(0, 5)
                random_choice = True
            else:
                chosen_q_index = np.argmax(q_values)  # select index of highest q-value

            # print('random choice:', random_choice)

            chosen_q = q_values[chosen_q_index]  # chosen q value
            chosen_direction = directions[chosen_q_index]  # choose corresponding direction

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
            if is_wall(next_cell):  # penalize wall hit
                self.wall_hit_count += 1
                print('hit wall at {},{}'.format(current_row, current_col))
                reward += self.rewards['wall']
                next_row, next_col = current_row, current_col  # revert position

            elif is_fire(next_cell):  # penalize fire hit
                reward += self.rewards['fire']
                print('hit fire at {},{}'.format(current_row, current_col))
                next_row, next_col = current_row, current_col  # revert position

            if (next_row, next_col) in self.visited:
                # todo - proportional to number of visitations?
                reward += self.rewards['revisited']  # * self.visited[(next_row, next_col)]

            distance_from_start = euclidian_cost(self.position, self.start_position)
            reward += self.rewards['distance_from_start'] * distance_from_start

            distance_to_end = euclidian_cost(self.position, self.end_position)
            reward += self.rewards['distance_to_end'] * distance_to_end

            if train:  # update q-table
                max_q = self.Q[next_row, next_col, :].max()
                current_q_value = self.Q[current_row, current_col, chosen_q_index]

                next_q_value = current_q_value + self.learning_rate * (reward - chosen_q + self.discount * max_q)
                # next_q_value = max(0, next_q_value)  # set minimum to 0

                self.Q[current_row, current_col, chosen_q_index] = next_q_value

                if self.step_count != 0:  # ignore initial step
                    if next_row == prev_row and next_col == prev_col:
                        self.Q[next_row, next_col, self.previous_q_index] += self.learning_rate * self.rewards[
                            'repeat_step']
                        # reward += self.rewards['repeat_step']  # penalize for repeating most recent step

            self.previous_position = np.copy(self.position)  # update memory of previous position
            self.previous_q_index = chosen_q_index

        self.position = np.array([next_row, next_col])  # update position
        self.step_count += 1

        self.visited[(current_row, current_col)] += 1  # increment visited nodes

        return is_finished

    def train(self,
              maze: np.ndarray,
              max_steps: int = 1_000_000  # maximum training steps per epoch
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
