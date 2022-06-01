import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import display.display as dp

from config import MAZE_PATH, ENABLE_FAST_READ_MAZE
from config import RANDOM_SEED
from config import EVAL_POSITION_HISTORY_COLOR, WALL_COLOR, PATH_COLOR, FIRE_COLOR

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
               log_file: str = None):

    eval_mazes = []
    step_counts = np.empty((num_epochs))  # store step_counts

    for epoch in range(num_epochs):

        # eval agent:
        eval_maze = rm.load_maze(MAZE_PATH)
        eval_maze = agent.eval(maze=eval_maze, max_steps=max_eval_steps)
        eval_mazes.append(eval_maze.copy())

        if log_eval:
            log_agent(agent, epoch=epoch, log_file_name=log_file)  # log full epoch history

        if plot:
            dp.plot_eval(agent=agent, epoch=epoch, maze=eval_maze)

        step_counts[epoch] = agent.step_count

    return step_counts, eval_mazes

if __name__ == '__main__':
    # todo - load pretrained weights
    # agent_pretrained = Agent()
    pass