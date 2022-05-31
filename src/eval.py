import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import display.display as dp

from config import MAZE_PATH, ENABLE_FAST_READ_MAZE
from config import NUMPY_SEED
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
               log_eval: bool = True):

    eval_mazes = []

    for epoch in range(num_epochs):

        # eval agent:
        eval_maze = rm.load_maze(MAZE_PATH)
        eval_maze = agent.eval(maze=eval_maze, max_steps=max_eval_steps)
        eval_mazes.append(eval_maze.copy())

        if log_eval:
            log_agent(agent, epoch=epoch)  # log full epoch history

        maze_walls = eval_maze[:, :, 0]
        maze_shape = maze_walls.shape

        if plot:
            fig, ax = plt.subplots()
            dp.plot_maze_walls(maze_walls,
                               ax=ax,
                               cmap=ListedColormap([PATH_COLOR, WALL_COLOR])
                               )

            eval_path = agent.history['position']
            dp.plot_agent_path(eval_path, shape=maze_shape,
                               ax=ax,
                               cmap=ListedColormap(['none', EVAL_POSITION_HISTORY_COLOR])
                               )

            ax.set_title('epoch={}, steps={}'.format(epoch, agent.step_count))

            plt.show()

    return eval_mazes

if __name__ == '__main__':
    # todo - load pretrained weights
    # agent_pretrained = Agent()
    pass