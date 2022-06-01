import src.mazes.sample_mazes

# agent classes:
from src.lib.log import log_agent
from agent.agent import Agent

from eval import eval_agent

import display.display as dp

import numpy as np

from config import MAZE_PATH, ENABLE_FAST_READ_MAZE, USE_OPTIMAL_POLICY, ENABLE_PATH_PLOTTING, ENABLE_RESULTS_PLOTTING
from config import TRAIN_MAX_STEPS, EVAL_MAX_STEPS, NUM_EPOCHS, NUM_EVAL_EPOCHS

if ENABLE_FAST_READ_MAZE:  # faster maze reading
    import lib.fast_read_maze as rm
else:
    import lib.read_maze as rm

# sample mazes:
walls_201 = rm.load_maze('./mazes/final.npy')[:, :, 0]
walls_9 = src.mazes.sample_mazes.sample_maze_9_A  # 9x9 test maze
walls_11 = src.mazes.sample_mazes.sample_maze_11_A


def train_optimal(agent: Agent,
                  num_epochs: int = NUM_EPOCHS,
                  max_eval_steps: int = EVAL_MAX_STEPS,
                  max_train_steps: int = TRAIN_MAX_STEPS,
                  eval: bool = True,  # evaluate
                  num_eval_epochs: int = NUM_EVAL_EPOCHS,  # evaluation epoch number
                  plot: bool = ENABLE_PATH_PLOTTING,  # plot path
                  log_train: bool = True,  # log history
                  log_eval: bool = True,
                  ):
    print('Starting Training Using Optimal Policy')

    train_mazes = []
    eval_mazes = []  # returns empty if eval disabled

    train_step_counts = np.empty(num_epochs)
    eval_step_counts = np.empty((num_epochs, num_eval_epochs))

    for epoch in range(num_epochs):
        # train agent:
        train_maze = rm.load_maze(MAZE_PATH)
        train_maze = agent.train(maze=train_maze, max_steps=max_train_steps)
        train_mazes.append(train_maze.copy())

        train_step_counts[epoch] = agent.step_count

        if log_train:
            log_agent(agent, epoch=epoch, log_file_name='train_optimal_epoch{}'.format(epoch))  # log full epoch history

        print('--- exited training epoch={}.'.format(epoch))

        if plot:
            dp.plot_training(agent=agent,
                             epoch=epoch,
                             maze=train_maze)

        print('plotted epoch={}'.format(epoch))

        if eval:  # evaluate after every epoch
            step_counts, eval_mazes = eval_agent(agent=agent,
                                                 max_eval_steps=max_eval_steps,
                                                 log_eval=log_eval,
                                                 log_file='eval_optimal_epoch{}'.format(epoch),
                                                 plot=plot,
                                                 num_epochs=num_eval_epochs)

            eval_step_counts[epoch, :] = step_counts

    return agent, train_step_counts, eval_step_counts


if __name__ == '__main__':

    maze = rm.load_maze(MAZE_PATH)
    maze_walls = maze[:, :, 0]  # walls for plotting shapes
    maze_shape = maze_walls.shape
    end_position = tuple(np.subtract(maze_shape, (2, 2)))  # calculate end position

    if USE_OPTIMAL_POLICY:
        optimal_agent = Agent(  # instantiate new agent
            end_position=end_position,
            height=maze_shape[1],
            width=maze_shape[0],
            n_actions=5,
        )

        optimal_agent, train_steps, eval_steps = train_optimal(optimal_agent)

        if ENABLE_RESULTS_PLOTTING:
            dp.plot_results(num_epochs=NUM_EPOCHS, num_eval_epochs=NUM_EVAL_EPOCHS,
                            train_steps=train_steps, eval_steps=eval_steps)
    else:
        # todo - implement alternative policies
        pass
