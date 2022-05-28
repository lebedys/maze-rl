import maze.mazes.sample_mazes
from maze.lib import read_maze
import agent

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import display.display as dp

import numpy as np

read_maze.load_maze('./mazes/final.npy')
walls_201 = read_maze.maze_cells[:, :, 0]

walls_9 = maze.mazes.sample_mazes.sample_maze_9_A  # 9x9 test maze
walls_11 = maze.mazes.sample_mazes.sample_maze_11_A

wall_color = '#eeeeee'
path_color = '#111111'

if __name__ == '__main__':

    maze_walls = walls_11
    maze_shape = (11, 11)
    end_position = (9, 9)

    a0 = agent.Agent(
        end_position=end_position
    )  # instantiate new agent

    train_path = a0.train(maze=maze_walls, max_steps=500)

    print('exited training.')

    fig, ax = plt.subplots()
    dp.plot_maze_walls(maze_walls,
                       ax=ax,
                       cmap=ListedColormap([path_color, wall_color])
                       )

    dp.plot_agent_path(train_path, shape=maze_shape,
                       ax=ax,
                       cmap=ListedColormap(['none', 'red'])
                       )

    plt.show()

    run_path = a0.run(maze=maze_walls, max_steps=10_000)

    fig2, ax = plt.subplots()
    dp.plot_maze_walls(maze_walls,
                       ax=ax,
                       cmap=ListedColormap([path_color, wall_color])
                       )

    dp.plot_agent_path(run_path, shape=maze_shape,
                       ax=ax,
                       cmap=ListedColormap(['none', 'blue'])
                       )

    plt.show()
