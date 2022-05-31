import src.mazes.sample_mazes

# agent classes:
from src.lib.log import log_agent
from agent.agent import Agent

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import display.display as dp

import numpy as np

from config import MAZE_PATH, ENABLE_FAST_READ_MAZE
from config import NUMPY_SEED
from config import TRAIN_POSITION_HISTORY_COLOR, EVAL_POSITION_HISTORY_COLOR, WALL_COLOR, PATH_COLOR, FIRE_COLOR

if ENABLE_FAST_READ_MAZE:  # faster maze reading
    import lib.fast_read_maze as rm
else:
    import lib.read_maze as rm

walls_201 = rm.load_maze('./mazes/final.npy')[:, :, 0]
walls_9 = src.mazes.sample_mazes.sample_maze_9_A  # 9x9 test maze
walls_11 = src.mazes.sample_mazes.sample_maze_11_A

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

    print('Starting Training')

    train_mazes = []
    eval_mazes = []

    for epoch in range(num_epochs):
        maze = rm.load_maze(MAZE_PATH)
        train_maze = a0.train(maze=maze, max_steps=max_steps)
        train_mazes.append(train_maze)

        log_agent(a0, epoch=epoch)  # log full epoch history

        print('exited training {}.'.format(epoch))

        fig, ax = plt.subplots()
        dp.plot_maze_walls(maze_walls,
                           ax=ax, cmap=ListedColormap([PATH_COLOR, WALL_COLOR]))

        dp.plot_agent_path(train_maze, shape=maze_shape,
                           ax=ax, cmap=ListedColormap(['none', TRAIN_POSITION_HISTORY_COLOR]))

        ax.set_title('epoch {}'.format(epoch))

        plt.show()

        print('plotted epoch {}'.format(epoch))

        # run without training for results:
        run_path = a0.run(maze=maze_walls, max_steps=100_000)

        fig, ax = plt.subplots()
        dp.plot_maze_walls(maze_walls,
                           ax=ax,
                           cmap=ListedColormap([PATH_COLOR, WALL_COLOR])
                           )

        dp.plot_agent_path(run_path, shape=maze_shape,
                           ax=ax,
                           cmap=ListedColormap(['none', EVAL_POSITION_HISTORY_COLOR])
                           )

        ax.set_title('epoch {}'.format(epoch))

        plt.show()

    return a0, train_mazes

if __name__ == '__main__':
    a0 = Agent(  # instantiate new agent
        end_position=end_position,
        height=maze_shape[1],
        width=maze_shape[0],
        n_actions=5,
    )

    trained_agent, training_paths = train_agent(a0,
                                                num_epochs=1,
                                                max_steps=1000)

# todo - penalize path length at end of epoch
#      - use this to update policy (q-table) somehow
