import display
import matplotlib.pyplot as plt
import numpy as np

import read_maze

if __name__ == '__main__':
    read_maze.load_maze('./mazes/final.npy')
    walls = read_maze.maze_cells[:, :, 0]

    fig, ax = plt.subplots(figsize=(10, 10))
    display.plot_maze_walls(1 - read_maze.maze_cells[:, :, 0], show_values=False, ax=ax)

    # agent position:
    a0 = (199, 199)
    display.plot_agent(a0, ax=ax)

    # agent path:
    path = np.random.randint(low=1, high=199, size=(100, 2))  # simulate (unrealistic) path
    display.plot_agent_path(path, ax=ax)

    plt.show()