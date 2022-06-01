import src.mazes.sample_mazes

# agent classes:
from src.lib.log import log_agent
from agent.agent import Agent

from eval import eval_agent

import display.display as dp

import numpy as np

from config import MAZE_PATH, ENABLE_FAST_READ_MAZE, USE_OPTIMAL_POLICY
from config import RANDOM_SEED

if ENABLE_FAST_READ_MAZE:  # faster maze reading
    import lib.fast_read_maze as rm
else:
    import lib.read_maze as rm


# sample mazes:
walls_201 = rm.load_maze('./mazes/final.npy')[:, :, 0]
walls_9 = src.mazes.sample_mazes.sample_maze_9_A  # 9x9 test maze
walls_11 = src.mazes.sample_mazes.sample_maze_11_A


def train_agent(agent: Agent,
                num_epochs: int = 1,
                max_eval_steps: int = 100_000,
                max_train_steps: int = 1_000_000,
                eval: bool = True,  # evaluate
                num_eval_epochs: int = 1,  # evaluation epoch number
                plot: bool = True,  # plot path
                log_train: bool = True,  # log history
                log_eval: bool = True
                ):
    print('Starting Training')

    train_mazes = []
    eval_mazes = []  # returns empty if eval disabled

    for epoch in range(num_epochs):
        # train agent:
        train_maze = rm.load_maze(MAZE_PATH)
        train_maze = a0.train(maze=train_maze, max_steps=max_train_steps)
        train_mazes.append(train_maze.copy())

        if log_train:
            log_agent(a0, epoch=epoch)  # log full epoch history

        print('--- exited training epoch={}.'.format(epoch))

        if plot:
            dp.plot_training(agent=a0,
                             epoch=epoch,
                             maze=train_maze)

        print('plotted epoch={}'.format(epoch))

        if eval:  # evaluate after every epoch
            eval_mazes = eval_agent(agent=a0,
                                    max_eval_steps=max_eval_steps,
                                    log_eval=log_eval,
                                    plot=plot,
                                    num_epochs=num_eval_epochs)

    return a0, train_mazes, eval_mazes


if __name__ == '__main__':

    maze = rm.load_maze(MAZE_PATH)
    maze_walls = maze[:, :, 0]  # walls for plotting shapes
    maze_shape = maze_walls.shape
    end_position = tuple(np.subtract(maze_shape, (2, 2)))  # calculate end position

    if USE_OPTIMAL_POLICY:
        pass  # todo
    else:

        a0 = Agent(  # instantiate new agent
            end_position=end_position,
            height=maze_shape[1],
            width=maze_shape[0],
            n_actions=5,
        )

        train_agent(a0, num_epochs=10,
                    max_train_steps=1000_000,
                    max_eval_steps=1000_000,
                    log_train=True,
                    log_eval=True
                    )