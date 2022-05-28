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
    'finish': 10,  # finishing maze

    # obstacles:
    'wall': -1.,  # hit wall
    'fire': -1.,  # hit fire

    # movement:
    'step': -0.1,  # take step in any direction
    'stay': -0.2,  # staying in place
    'return': -0.5,  # take step backwards
    'visited': 0,  # todo - penalize return to visited position
}


class Agent:

    def __init__(self,
                 start_position: Tuple[int, int] = (1, 1),  # initial position
                 end_position: Tuple[int, int] = (199, 199),  # final position
                 max_steps: int = 10_000_000,   # maximum path length
                 Q: np.ndarray = None,          # Q-table
                 rewards: dict = REWARDS,       # rewards
                 discount: float = 0.1,         # discount factor
                 euclidian_cost_weighting: float = 0.0  # todo - explore impact of this parameter
                 ):

        self.start_position = np.array(list(start_position))
        self.end_position = np.array(list(end_position))
        self.Q = Q
        self.rewards = rewards
        self.discount = discount  # discount factor (gamma)

        self.euclidian_cost_weighting = euclidian_cost_weighting

        # initialize agent position:
        self.position = start_position  # initial position

        if Q is None:  # if no predefined Q-table provided
            self.Q = np.full((WIDTH, HEIGHT, N_ACTIONS), 1. / N_ACTIONS)  # initialize default q values
            # todo - add noise or some randomness

    def reset_position(self) -> None:
        self.position = self.start_position  # initial position

    def is_finished(self) -> bool:
        return self.position == self.end_position

    def observe(self, maze: np.ndarray) -> np.ndarray:
        # [[0, 1, 2],
        #  [3, 4, 5],
        #  [6, 7, 8]]

        x, y = self.position
        return maze[x - 1:x + 1, y - 1:y + 1]

    def step(self,
             observation: np.ndarray,  # local observation
             train=False,  # enable training
             epsilon: float = 0.,  # todo - exploration
             q_noise: float = 0.,  # todo - random noise
             ) -> None:  # random exploration probability

        # todo - epsilon for random choice selection
        # todo - add random noise to q values
        # todo - invalidate some directions

        if np.random.random() < epsilon:  # random exploration
            chosen_direction = np.random.choice(directions)
        else:
            # get Q(s) vector:
            q_values = self.Q[self.position[0], self.position[1], :]

            # todo - random noise applied to q value vector

            # pick maximum q-value from Q(s):
            max_q = np.max(q_values)  # maximum q value
            chosen_direction = directions[np.argmax(q_values)]  # select corresponding direction

        reward = self.rewards['step']

        # flatten observation: [0, 1, 2, 3, 4, 5, 6, 7, 8]
        walls, fires = observation
        walls = walls.flatten()
        fires = fires.flatten()

        next_position = self.position  # initialize next position calculation
        if chosen_direction == Direction.NONE:  # no change in position
            reward += self.rewards['stay']  # penalty for staying in place
        elif chosen_direction == Direction.UP:
            next_position[1] -= 1  # move up
        elif chosen_direction == Direction.RIGHT:
            next_position[0] += 1  # move right
        elif chosen_direction == Direction.DOWN:
            next_position[1] += 1  # move down
        elif chosen_direction == Direction.LEFT:
            next_position[1] -= 1  # move left
        else:
            raise ValueError()  # fixme - correct exception type needed

        # do not allow walking through obstacles:
        if is_wall(chosen_direction, walls):  # penalize wall hit
            reward += self.rewards['wall']
        elif is_fire(chosen_direction, fires):  # penalize fire hit
            reward += self.rewards['fire']
        else:
            self.position = next_position

        previous_position = 0
        if next_position == previous_position:
            pass  # todo - check and penalize if returning to previous move

        # small negative reward for distance to goal
        reward += euclidian_cost(self.position, self.end_position) * self.euclidian_cost_weighting

        # todo - update qtable

    def train(self,
              maze: np.ndarray,
              max_steps: int = 10  # maximum training steps per epoch
              ) -> np.ndarray:

        self.reset_position()  # initialize agent

        # initialize training path:
        train_path = np.empty(max_steps, 2)  # empty path
        train_path[0, :] = self.position  # add initial position

        for i in range(max_steps):
            observation = self.observe(maze)  # get surrounding information

            self.step(observation, train=True)  # update position
            train_path[i] = self.position  # store new position

            if self.is_finished():  # achieved goal?
                train_path = train_path[:i+1, :]  # truncate if path unfilled before returning
                break  # end training

        return train_path  # return path when finished

    def run(self, maze, max_steps=10):  # run pretrained agent
        self.reset_position()  # initialize agent

        run_path = np.empty(max_steps, 2)
        run_path[0, :] = self.position  # add initial position

        for i in range(max_steps):
            self.step(maze, train=False)

            run_path[i, :] = self.position  # append position

            if self.is_finished():  # achieved goal?
                run_path = run_path[:i+1, :]  # truncate path if unfilled before returning
                break  # end run

        return run_path  # return path when finished