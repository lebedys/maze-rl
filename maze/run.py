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

maze_walls = walls_201
maze_shape = (201, 201)
end_position = (199, 199)

def train(num_epochs: int = 1):

    a0 = agent.Agent(
        end_position=end_position
    )  # instantiate new agent

    for epoch in range(num_epochs):

        train_path = a0.train(maze=maze_walls, max_steps=1000_000)

        print('exited training {}.'.format(epoch))

        fig, ax = plt.subplots()
        dp.plot_maze_walls(maze_walls,
                           ax=ax,
                           cmap=ListedColormap([path_color, wall_color])
                           )

        dp.plot_agent_path(train_path, shape=maze_shape,
                           ax=ax,
                           cmap=ListedColormap(['none', 'orange'])
                           )

        ax.set_title('epoch {}'.format(epoch))

        plt.show()


if __name__ == '__main__':

    train(num_epochs=10)

    # train_path = a0.train(maze=maze_walls, max_steps=1000_000)
    #
    # print('exited training.')
    #
    # fig, ax = plt.subplots()
    # dp.plot_maze_walls(maze_walls,
    #                    ax=ax,
    #                    cmap=ListedColormap([path_color, wall_color])
    #                    )
    #
    # dp.plot_agent_path(train_path, shape=maze_shape,
    #                    ax=ax,
    #                    cmap=ListedColormap(['none', 'red'])
    #                    )
    #
    # plt.show()
    #
    # # todo - penalize path length at end of epoch
    # #      - use this to update policy (q-table) somehow
    #
    # run_path = a0.run(maze=maze_walls, max_steps=10_000)
    #
    # fig2, ax = plt.subplots()
    # dp.plot_maze_walls(maze_walls,
    #                    ax=ax,
    #                    cmap=ListedColormap([path_color, wall_color])
    #                    )
    #
    # dp.plot_agent_path(run_path, shape=maze_shape,
    #                    ax=ax,
    #                    cmap=ListedColormap(['none', 'blue'])
    #                    )
    #
    # plt.show()

