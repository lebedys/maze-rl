import src.mazes.sample_mazes
from lib import read_maze

from agent.agent import Agent
from agent.log import log_agent

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import display.display as dp

import numpy as np

from config import *

read_maze.load_maze('./mazes/final.npy')
walls_201 = read_maze.maze_cells[:, :, 0]

walls_9 = src.mazes.sample_mazes.sample_maze_9_A  # 9x9 test maze
walls_11 = src.mazes.sample_mazes.sample_maze_11_A

wall_color = '#eeeeee'
path_color = '#111111'

maze_walls = walls_201
maze_shape = maze_walls.shape
end_position = tuple(np.subtract(maze_shape, (2, 2)))

def eval_agent(agent: Agent,
              num_epochs: int = 1,
              max_steps: int = 100_000):
    # todo - return evaluated agent
    return []


def train_agent(agent: Agent,
                num_epochs: int = 1,
                max_steps: int = 1_000_000,
                eval: bool = True,  # evaluate
                plot: bool = True,  # plot path
                log: bool = True,   # log history
                ):

    train_paths = []

    for epoch in range(num_epochs):
        train_path = a0.train(maze=maze_walls, max_steps=max_steps)
        train_paths.append(train_path)

        log_agent(a0, epoch=epoch)

        print('exited training {}.'.format(epoch))

        fig, ax = plt.subplots()
        dp.plot_maze_walls(maze_walls,
                           ax=ax, cmap=ListedColormap([path_color, wall_color]))

        dp.plot_agent_path(train_path, shape=maze_shape,
                           ax=ax, cmap=ListedColormap(['none', 'orange']))

        ax.set_title('epoch {}'.format(epoch))

        plt.show()

        print('plotted epoch {}'.format(epoch))

        # run without training for results:
        run_path = a0.run(maze=maze_walls, max_steps=100_000)

        fig, ax = plt.subplots()
        dp.plot_maze_walls(maze_walls,
                           ax=ax,
                           cmap=ListedColormap([path_color, wall_color])
                           )

        dp.plot_agent_path(run_path, shape=maze_shape,
                           ax=ax,
                           cmap=ListedColormap(['none', 'blue'])
                           )

        ax.set_title('epoch {}'.format(epoch))

        plt.show()

    return a0, train_paths


np.random.seed(2022)

if __name__ == '__main__':
    a0 = Agent(  # instantiate new agent
        end_position=end_position,
        height=maze_shape[1],
        width=maze_shape[0],
        n_actions=5,
    )

    trained_agent, training_paths = train_agent(a0, num_epochs=10, max_steps=1_000_000)

# todo - penalize path length at end of epoch
#      - use this to update policy (q-table) somehow
