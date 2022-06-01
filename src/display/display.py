import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.config import PATH_COLOR, WALL_COLOR, TRAIN_POSITION_HISTORY_COLOR, EVAL_POSITION_HISTORY_COLOR
from src.agent.agent import Agent


def plot_training(agent: Agent, epoch: int, maze: np.ndarray):
    # if ax is None:
    #     ax = plt.gca()  # get current axis

    maze_walls = maze[:, :, 0]
    maze_shape = maze_walls.shape

    fig, ax = plt.subplots()
    plot_maze_walls(maze_walls,
                    ax=ax, cmap=ListedColormap([PATH_COLOR, WALL_COLOR]))

    train_path = agent.history['position']
    plot_agent_path(train_path, shape=maze_shape,
                    ax=ax, cmap=ListedColormap(['none', TRAIN_POSITION_HISTORY_COLOR]))

    ax.set_title('TRAINING - epoch={}, steps={}'.format(epoch, agent.step_count))

    plt.show()


def plot_eval(agent: Agent, epoch: int, maze: np.ndarray):
    maze_walls = maze[:, :, 0]
    maze_shape = maze_walls.shape

    fig, ax = plt.subplots()
    plot_maze_walls(maze_walls,
                    ax=ax,
                    cmap=ListedColormap([PATH_COLOR, WALL_COLOR])
                    )

    eval_path = agent.history['position']
    plot_agent_path(eval_path, shape=maze_shape,
                    ax=ax,
                    cmap=ListedColormap(['none', EVAL_POSITION_HISTORY_COLOR])
                    )

    ax.set_title('EVALUATION - steps={steps}'.format(steps=agent.step_count))

    plt.show()


def plot_maze_walls(walls, show_values=False, ax=None, cmap=None) -> None:
    if ax is None:
        ax = plt.gca()  # get current axis

    # display walls
    if cmap is None:
        wall_color = '#eeeeee'
        path_color = '#111111'
        cmap = ListedColormap([wall_color, path_color])

    ax.matshow(walls, cmap=cmap)

    # if show_values:
    for (row, col), value in np.ndenumerate(walls):
        if show_values:
            ax.text(col, row, '{:1d}'.format(value), ha='center', va='center',
                    bbox=dict(boxstyle='round', edgecolor='none', facecolor=wall_color))


def plot_agent(position, ax=None):
    if ax is None:
        ax = plt.gca()  # get current axis

    agent = plt.Circle(position, 0.2, color='blue')
    ax.add_patch(agent)

    range = plt.Rectangle((position[0] - 1.5, position[1] - 1.5), 3, 3, ec="blue", facecolor='none')
    ax.add_patch(range)

    # todo - add t label option


def plot_agent_path(path: np.ndarray, shape: np.ndarray, ax=None, cmap=None):
    if ax is None:
        ax = plt.gca()  # get current axis

    # display walls
    if cmap is None:
        none = 'none'
        path_color = 'red'
        cmap = ListedColormap([none, path_color])

    # convert to mask matrix:
    path_mat = np.zeros(shape=shape)

    rows = path[:, 0].astype(int)
    cols = path[:, 1].astype(int)
    visited = np.ones_like(rows)

    path_mat[rows, cols] = visited

    ax.matshow(path_mat, cmap=cmap)


def plot_fires(agent_position: tuple, agent_observation, ax=None):
    if ax is None:
        ax = plt.gca()  # get current axis

    walls, fires = agent_observation[:, :, 0], agent_observation[:, :, 1]

    # if show_values:
    for (row, col), value in np.ndenumerate(walls):
        fire_time = walls[row, col]

        if fire_time == 0 or (col, row) == agent_position:
            pass  # do nothing
        else:
            if fire_time == 1:
                fire_color = 'orange'
            elif fire_time == 2:
                fire_color = 'red'

            # todo - draw
            ax.text(col, row,
                    '{:1d}'.format(fires[row, col]),
                    ha='center', va='center',
                    size='x-large',
                    bbox=dict(
                        boxstyle='square',
                        edgecolor='none',
                        facecolor=fire_color,
                        alpha=0.5)
                    )
