from maze.lib import read_maze
import agent

import numpy as np

read_maze.load_maze('./mazes/final.npy')
walls = read_maze.maze_cells[:, :, 0]

if __name__ == '__main__':
    a0 = agent.Agent()  # instantiate new agent
    p0 = a0.run(walls)

    # todo - plot untrained agent path

    a1 = a0.train()  # todo - train agent with one run
    p1 = a1.run()  # todo - run trained agent
