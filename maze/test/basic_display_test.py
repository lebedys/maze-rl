from maze.display import display
import matplotlib.pyplot as plt
import numpy as np

import maze.lib.read_maze as rm

if __name__ == '__main__':
    rm.load_maze('./mazes/final.npy')
    walls = rm.maze_cells[:, :, 0]

    fig, ax = plt.subplots(figsize=(10, 10))
    display.plot_maze_walls(1 - rm.maze_cells[:, :, 0], show_values=False, ax=ax)

    # agent position:
    a0 = (199, 199)
    display.plot_agent(a0, ax=ax)

    # agent path:
    path = np.random.randint(low=1, high=199, size=(100, 2))  # simulate (unrealistic) path
    display.plot_agent_path(path, ax=ax)

    plt.show()