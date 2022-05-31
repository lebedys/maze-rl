import numpy as np
from typing import Tuple
from collections import defaultdict
from enum import IntEnum

from src.lib.util import is_fire, is_wall, euclidian_cost

from src.config import *

if ENABLE_FAST_READ_MAZE:  # faster maze reading
    import src.lib.fast_read_maze as rm
else:
    import src.lib.read_maze as rm


# todo - implement periodic logging

# directionality:
# [0, 1, 2
#  3, 4, 5
#  6, 7, 8]

class Direction(IntEnum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

    NONE = 0


reverse_direction_mapping = {
    Direction.NONE: Direction.NONE,
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.RIGHT: Direction.LEFT,
    Direction.LEFT: Direction.RIGHT
}


def get_reverse_direction(direction: Direction):
    return reverse_direction_mapping[direction]


class Agent:

    def __init__(self,
                 start_position: Tuple[int, int] = (1, 1),  # initial position
                 end_position: Tuple[int, int] = (199, 199),  # final position

                 # parameters:
                 Q: np.ndarray = None,  # Q-table
                 width: int = 201,
                 height: int = 201,
                 n_actions: int = 5,

                 # rewards:
                 rewards: dict = REWARDS,  # rewards

                 # hyper-parameters:
                 learning_rate: float = 0.3,  # q-table learning rate
                 discount: float = 0.9,  # discount factor
                 exploration_epsilon: float = 0.  # probability of random exploration
                 ):

        self.start_position = np.array(list(start_position), dtype=int)
        self.end_position = np.array(list(end_position), dtype=int)
        self.Q = Q
        self.rewards = rewards

        # hyper-parameters
        self.learning_rate = learning_rate
        self.discount = discount  # discount factor (gamma)
        self.exploration_epsilon = exploration_epsilon  # probability of random exploration

        # initialize agent position:
        self.position: np.ndarray = np.copy(self.start_position)  # initial position

        # logging parameters:
        self.step_count: int = 0
        self.wall_hit_count: int = 0
        self.fire_hit_count: int = 0

        # hashmap of visited nodes:
        self.visited = defaultdict(int)

        # memory of previous step:
        self.previous_position = np.array([0, 0])
        self.previous_q_index = Direction.NONE

        # todo - replay buffer:
        #           - store N previous steps in stack
        #           - penalize backtracking through rollback

        # empty history:
        self.history = {
            'position': None,
            'observation': None,
            'q_values': None,
            'action': None,
            'is_random_choice': None,
            'next_position': None,
            'is_finished': None,
            'learning_rate': None,
            'discount_factor': None,
            'epsilon': None,
            'reverse_learning_rate': None,
            'reverse_discount_factor': None
        }

        if self.Q is None:  # if no Q-table provided
            self.init_q_table(width, height, n_actions)

    def init_q_table(self, width: int, height: int, n_actions: int) -> None:
        self.Q = np.full((width, height, n_actions), 0.)

        # random normal initial values:
        # self.Q = 0.5 * np.random.normal(loc=0.5, scale=0.1, size=(WIDTH, HEIGHT, N_ACTIONS))

        # initial position allows only RIGHT and DOWN directions
        self.Q[1, 1, :] = np.array([
            -np.inf,  # NONE
            -np.inf,  # UP
            0, # np.random.normal(loc=0.5, scale=0.1),  # RIGHT
            0, # np.random.normal(loc=0.5, scale=0.1),  # DOWN
            -np.inf,  # LEFT
        ])

    def reset_history(self, max_steps: int) -> None:
        # run_path = np.empty(shape=(max_steps + 1, 2), dtype=int)
        # run_path[0, :] = self.position  # add initial position

        self.history = {
            'position': np.empty(shape=(max_steps, 2), dtype=int),
            'observation': np.empty(shape=(max_steps, 3, 3, 2), dtype=int),
            'q_values': np.empty(shape=(max_steps, 5), dtype=float),
            'action': np.empty(shape=(max_steps), dtype=Direction),
            'is_random_choice': np.empty(shape=(max_steps), dtype=bool),
            'next_position': np.empty(shape=(max_steps, 2), dtype=int),
            'is_finished': np.empty(shape=(max_steps), dtype=bool),
            'learning_rate': np.empty(shape=(max_steps), dtype=bool),
            'discount_factor': np.empty(shape=(max_steps), dtype=bool),
            'epsilon': np.empty(shape=(max_steps), dtype=bool),
            'reverse_learning_rate': np.empty(shape=(max_steps), dtype=bool),
            'reverse_discount_factor': np.empty(shape=(max_steps), dtype=bool),
        }

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

        # todo - log observation

        row, col = self.position
        new_maze, observation = rm.get_local_maze_information(maze, row, col)
        # return maze[row - 1: (row + 1) + 1, col - 1: (col + 1) + 1]
        return new_maze, observation

    def invalidate_walls(self, q_values: np.ndarray, walls: np.ndarray, fires: np.ndarray) -> np.ndarray:

        walls = np.array([0.0,  # invalidate Direction.NONE
                          walls[0, 1],  # invalidate all walls
                          walls[1, 2],
                          walls[2, 1],
                          walls[1, 0]
                          ])

        q_values[walls == 0.0] = -np.inf
        return q_values

    def step(self,
             walls: np.ndarray,  # local observation of walls
             fires: np.ndarray,  # local observation of fires
             train: bool = False,  # enable training (updating Q-table)

             # hyper-parameters:
             exploration_epsilon: float = 0.,
             learning_rate: float = 0.1,
             discount_factor: float = 0.9,

             # reverse hyper-parameters:
             reverse_learning_rate: float = 0.1,
             reverse_discount_factor: float = 0.9
             ) -> bool:  # random exploration probability

        reward = self.rewards['step_taken']  # initialize reward

        # decompose positions into row, col
        # prev_row, prev_col = self.previous_position  # previous position
        current_row, current_col = self.position  # current position

        # -----------------------
        # check if maze completed
        if self.is_finished():
            # todo
            self.history['is_finished'][self.step_count] = True
            return True

        # select Q(s) vector:
        q_values = self.Q[current_row, current_col, :]

        # consider only VALID ACTIONS:
        q_values = self.invalidate_walls(q_values, walls, fires)
        # todo - temporarily invalidate fires somehow

        is_random_choice = False
        if np.random.random() < self.exploration_epsilon:  # random exploration
            is_random_choice = True
            while True:  # only select if not -np.inf
                chosen_q_index = np.random.randint(0, 5)
                if q_values[chosen_q_index] != -np.inf:
                    break
        else:
            max_q_indeces = np.argwhere(q_values == np.amax(q_values))
            max_q_indeces = [e[0] for e in max_q_indeces]
            chosen_q_index = np.random.choice(max_q_indeces)  # select index of highest q-value
            # chosen_q_index = np.argmax(q_values)

        chosen_direction = Direction(chosen_q_index)  # choose corresponding direction
        reverse_chosen_direction = get_reverse_direction(chosen_direction)
        chosen_q = q_values[chosen_direction]  # chosen q value

        next_row, next_col = current_row, current_col
        if chosen_direction == Direction.NONE:  # no change in position
            reward += self.rewards['stay']  # penalty for staying in place
            next_cell = walls[1, 1]
            next_fire = fires[1, 1]
        elif chosen_direction == Direction.UP:
            next_row -= 1  # move up
            next_cell = walls[0, 1]
            next_fire = fires[0, 1]
        elif chosen_direction == Direction.RIGHT:
            next_col += 1  # move right
            next_cell = walls[1, 2]
            next_fire = fires[1, 2]
        elif chosen_direction == Direction.DOWN:
            next_row += 1  # move down
            next_cell = walls[2, 1]
            next_fire = fires[2, 1]
        elif chosen_direction == Direction.LEFT:
            next_col -= 1  # move left
            next_cell = walls[1, 0]
            next_fire = fires[1, 0]
        else:
            raise ValueError('Directionality value not valid.')

        # reward function:
        if is_fire(next_fire):  # penalize fire hit
            # reward += self.rewards['fire']
            # print('hit fire at {},{}'.format(current_row, current_col))
            next_row, next_col = current_row, current_col  # revert position
            # reward = self.rewards['fire']
            # todo - better fire policy
        else:

            if (np.array([next_row, next_col]) == self.end_position).all():
                reward = self.rewards['finish']
            else:
                # penalize hitting obstacles:
                if is_wall(next_cell):  # penalize wall hit
                    self.wall_hit_count += 1
                    # print('hit wall at {},{}'.format(current_row, current_col))
                    reward += self.rewards['wall']
                    next_row, next_col = current_row, current_col  # revert position

                if is_fire(next_fire):  # penalize fire hit
                    # reward += self.rewards['fire']
                    # print('hit fire at {},{}'.format(current_row, current_col))
                    next_row, next_col = current_row, current_col  # revert position

                    # todo - consider waiting here

                # re-visitation penalties:
                if (next_row, next_col) in self.visited:
                    # penalty is inversely proportional to number of cell re-visitations
                    reward += self.rewards['revisited'] / self.visited[(next_row, next_col)]
                else:
                    reward += self.rewards['unvisited']

                # distance penalties:
                distance_from_start = euclidian_cost(self.position, self.start_position)
                reward += self.rewards['distance_from_start'] * distance_from_start

                distance_to_end = euclidian_cost(self.position, self.end_position)
                reward += self.rewards['distance_to_end'] * distance_to_end

            # -------------
            # TRAIN Q-TABLE
            # -------------
            if train:  # update q-table
                max_q = self.Q[next_row, next_col, :].max()  # todo - what to do if finished?
                current_q_value = self.Q[current_row, current_col, chosen_q_index]

                new_q_value = current_q_value + learning_rate * (reward + self.discount * max_q - current_q_value)
                self.Q[current_row, current_col, chosen_q_index] = new_q_value  # perform update

                # punish any backtracking:
                reverse_reward = self.rewards['repeat_step']

                reverse_q_i = reverse_chosen_direction
                reverse_max_q = self.Q[current_row, current_col, :].max()
                reverse_q_value = self.Q[next_row, next_col, reverse_chosen_direction]
                new_reverse_q_value = reverse_q_value + reverse_learning_rate * (reverse_reward + reverse_discount_factor * reverse_max_q - reverse_q_value)
                self.Q[next_row, next_col, reverse_chosen_direction] = new_reverse_q_value

        # ---------------
        #   LOG HISTORY
        # ---------------

        # action
        self.history['action'][self.step_count] = chosen_direction
        self.history['is_random_choice'][self.step_count] = is_random_choice
        self.history['next_position'][self.step_count] = np.array([next_row, next_col])

        # hyper-parameters
        self.history['learning_rate'][self.step_count] = learning_rate
        self.history['discount_factor'][self.step_count] = discount_factor
        self.history['epsilon'][self.step_count] = exploration_epsilon
        self.history['reverse_learning_rate'][self.step_count] = reverse_learning_rate
        self.history['reverse_discount_factor'][self.step_count] = reverse_discount_factor

        # ---------------
        # UPDATE POSITION
        # ---------------

        # self.previous_position = np.copy(self.position)  # update memory of previous position
        # self.previous_q_index = chosen_q_index

        self.position = np.array([next_row, next_col])  # update position

        self.step_count += 1
        self.visited[(current_row, current_col)] += 1  # increment visited nodes

        return False

    def truncate_history(self, final_steps):  # truncate if finished before max_steps
        # todo - truncate all
        self.history['position'] = self.history['position'][:final_steps + 1, :]
        self.history['observation'] = self.history['observation'][:final_steps + 1, :, :, :]
        self.history['q_values'] = self.history['q_values'][:final_steps + 1, :]
        self.history['action'] = self.history['action'][:final_steps + 1]
        self.history['is_random_choice'] = self.history['is_random_choice'][:final_steps + 1]
        self.history['is_finished'] = self.history['is_finished'][:final_steps + 1]

    def run(self,
            maze: np.ndarray,
            train: bool = True,  # enable training
            max_steps: int = 100_000):  # run agent

        self.reset_position()  # initialize position
        self.reset_history(max_steps=max_steps)  # initialize history

        for step in range(max_steps):
            self.history['position'][step, :] = self.position  # store current position

            maze, observation = self.observe(maze)  # get surrounding information
            walls = observation[:, :, 0]
            fires = observation[:, :, 1]

            self.history['observation'][step, :, :, :] = observation  # store current observation

            is_finished = self.step(walls=walls, fires=fires, train=train)  # update position

            self.history['is_finished'][step] = is_finished
            # run_path[step + 1, :] = np.copy(self.position)  # store new position

            if is_finished:  # achieved goal?
                # todo - add final position (with N/A for other values)
                self.truncate_history(step)
                print('finished running in {steps} steps!'.format(steps=self.step_count))
                break  # end training

        print('run loop broken after {} steps.'.format(self.step_count))

        return maze  # return path when finished

    def eval(self, maze: np.ndarray,
             max_steps: int = 100_000,  # maximum evaluation steps per epoch
             ):

        final_maze = self.run(maze, train=False, max_steps=max_steps)

        return final_maze

    def train(self, maze: np.ndarray,
              max_steps: int = 1_000_000,  # maximum training steps per epoch
              ):

        final_maze = self.run(maze, train=True, max_steps=max_steps)

        return final_maze

    # def train(self,
    #           maze: np.ndarray,
    #           max_steps: int = 1_000_000
    #           ) -> np.ndarray:
    #
    #     self.reset_position()  # initialize agent
    #
    #     # initialize training path:
    #     train_path = np.empty(shape=(max_steps + 1, 2), dtype=int)  # empty path
    #     train_path[0, :] = np.copy(self.position)  # add initial position
    #
    #     for i in range(max_steps):
    #         observation = self.observe(maze)  # get surrounding information
    #         walls = observation
    #         fires = []
    #         # walls, fires = observation[:, :, 0], observation[:, :, 1]
    #
    #         is_finished = self.step(walls=walls, fires=fires, train=True)  # update position
    #         train_path[i + 1, :] = np.copy(self.position)  # store new position
    #
    #         if is_finished:  # achieved goal?
    #             train_path = train_path[:i + 2, :]  # truncate if path unfilled before returning
    #             print('finished training in {} steps!'.format(self.step_count))
    #             break  # end training
    #
    #     print('training loop broken after {} steps.'.format(self.step_count))
    #
    #     return train_path  # return path when finished
