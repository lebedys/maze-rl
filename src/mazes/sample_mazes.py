import numpy as np
from src.display import display as dp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# todo - make mazes

sample_maze_9_A = np.array([
    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 1., 1., 0.],
    [0., 1., 0., 0., 0., 0., 0., 1., 0.],
    [0., 1., 1., 0., 1., 0., 1., 1., 0.],
    [0., 1., 0., 0., 1., 0., 1., 0., 0.],
    [0., 1., 1., 1., 1., 0., 1., 1., 0.],
    [0., 0., 1., 0., 1., 0., 0., 1., 0.],
    [0., 1., 1., 0., 1., 0., 1., 1., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0.]
])

sample_maze_11_A = np.array([
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 1., 1., 1., 0., 1., 1., 1., 0.],
    [0., 1., 0., 1., 0., 1., 1., 1., 0., 1., 0.],
    [0., 1., 1., 1., 0., 1., 0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
    [0., 1., 1., 1., 0., 1., 0., 0., 0., 1., 0.],
    [0., 1., 0., 1., 1., 1., 1., 1., 0., 1., 0.],
    [0., 1., 0., 1., 0., 0., 0., 1., 1., 1., 0.],
    [0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 0., 1., 0., 1., 1., 1., 1., 1., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
])

wall_color = '#eeeeee'
path_color = '#111111'

if __name__ == '__main__':
    fig, ax = plt.subplots()
    dp.plot_maze_walls(sample_maze_9_A,
                       ax=ax,
                       cmap=ListedColormap([path_color, wall_color])
                       )

    ax.set_title('Sample 9x9 Maze')
    plt.show()

    fig1, ax = plt.subplots()
    dp.plot_maze_walls(sample_maze_11_A,
                       ax=ax,
                       cmap=ListedColormap([path_color, wall_color])
                       )

    ax.set_title('Sample 11x11 Maze')
    plt.show()
