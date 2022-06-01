import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import display.display as dp

from config import MAZE_PATH, ENABLE_FAST_READ_MAZE
from config import RANDOM_SEED
from config import PRETRAINED_Q_PATH

if ENABLE_FAST_READ_MAZE:  # faster maze reading
    import lib.fast_read_maze as rm
else:
    import lib.read_maze as rm

# agent classes:
from src.lib.log import log_agent
from agent.agent import Agent


def eval_agent(agent: Agent,
               num_epochs: int = 1,
               max_eval_steps: int = 100_000,
               plot: bool = True,
               log_eval: bool = True,
               log_file: str = None,
               training_epoch: int = 0):
    eval_mazes = []
    step_counts = np.empty((num_epochs))  # store step_counts

    for epoch in range(num_epochs):

        # eval agent:
        eval_maze = rm.load_maze(MAZE_PATH)
        eval_maze = agent.eval(maze=eval_maze, max_steps=max_eval_steps)
        eval_mazes.append(eval_maze.copy())

        if log_eval:
            log_agent(agent, epoch=training_epoch,
                      log_file_name='{root}_{epoch}'.format(root=log_file, epoch=epoch))  # log full epoch history

        if plot:
            dp.plot_eval(agent=agent, epoch=epoch, maze=eval_maze)

        step_counts[epoch] = agent.step_count

    return step_counts, eval_mazes


def eval_pretrained(file_path: str) -> None:
    # todo - use pretrained weights
    agent_pretrained = Agent()


if __name__ == '__main__':

    abs_pretrained_q_path = os.path.abspath(PRETRAINED_Q_PATH)

    try:
        maze = np.load(abs_pretrained_q_path, allow_pickle=False, fix_imports=True)
    except OSError:
        print('File {f} does not exist'.format(f=abs_pretrained_q_path))
