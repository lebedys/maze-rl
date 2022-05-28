import maze.mazes.sample_mazes
from maze.lib import read_maze
import agent

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import display.display as dp

import numpy as np

read_maze.load_maze('./mazes/final.npy')
walls = read_maze.maze_cells[:, :, 0]

walls_9 = maze.mazes.sample_mazes.sample_maze_9_A  # 9x9 test maze

wall_color = '#eeeeee'
path_color = '#111111'

if __name__ == '__main__':

    a0 = agent.Agent(
        end_position=(7, 7)
    )  # instantiate new agent

    train_path = a0.train(maze=walls_9, max_steps=300)

    fig, ax = plt.subplots()
    dp.plot_maze_walls(walls_9,
                       ax=ax,
                       cmap=ListedColormap([path_color, wall_color])
                       )

    dp.plot_agent_path(train_path, shape=(9, 9),
                       ax=ax,
                       cmap=ListedColormap(['none', 'red'])
                       )

    plt.show()
