import os
from datetime import datetime

import numpy as np

from src.agent.agent import Agent, Direction


def cell_to_str(cell: np.ndarray) -> str:
    cell_str = ' '
    if cell[0] == 0.0:
        cell_str = 'W'  # wall
    elif cell[1] > 0:
        cell_str = str(int(cell[1]))  # fire
    return cell_str


def observation_to_str(observation: np.ndarray) -> str:
    # todo - parametrize for any observation size
    # todo - parse and beautify
    observation_str = '''
                        |{a_00}|{a_01}|{a_02}|
                        |{a_10}|{a_11}|{a_12}|
                        |{a_20}|{a_21}|{a_22}|
    '''.format(
        a_11='X',  # agent
        # observation cells:
        a_00=cell_to_str(observation[0, 0, :]),
        a_01=cell_to_str(observation[0, 1, :]),
        a_02=cell_to_str(observation[0, 2, :]),
        a_10=cell_to_str(observation[1, 0, :]),
        a_12=cell_to_str(observation[1, 2, :]),
        a_20=cell_to_str(observation[2, 0, :]),
        a_21=cell_to_str(observation[2, 1, :]),
        a_22=cell_to_str(observation[2, 2, :]),
    )
    return observation_str


def params_to_str(agent: Agent) -> str:
    params_str = '''
    rewards: {rewards}
    '''.format(
        rewards=agent.rewards
    )

    return params_str


def hyper_params_to_str(agent: Agent) -> str:
    hyper_params_str = '''
    learning rate: {lr},
    exploration factor (epsilon): {epsilon}
    '''.format(
        lr=agent.learning_rate,
        epsilon=agent.exploration_epsilon
        # is_training=agent.train
    )

    # todo - log max_steps
    # todo - log training_enabled

    return hyper_params_str


def history_to_str(agent: Agent) -> str:
    history_str = ''

    for step in range(agent.step_count):
        row, col = agent.history['position'][step]
        observation_str = observation_to_str(agent.history['observation'][step])
        q_values = agent.history['q_values'][step]

        try:
            action = agent.history['action'][step].name
        except BaseException:  # in case not within Enum range
            action = agent.history['action'][step]

        is_random_choice = agent.history['is_random_choice'][step]
        is_finished = agent.history['is_finished'][step]

        history_str += '''
        ---
        step = {step},
        position = ({row}, {col}),
        observation = {observation},
        q_values = {q_values},
        chosen action = {action},
        random exploration choice? = {is_random_choice},
        maze finished? = {is_finished}
        ---
        
        '''.format(
            step=step,
            row=row, col=col,
            observation=observation_str,
            q_values=q_values,
            action=action,
            is_random_choice=is_random_choice,
            is_finished=is_finished
        )

    return history_str


# todo - allow logging to chosen file
def log_agent(agent: Agent, epoch: int,
              log_dir: str = 'log/',
              log_file_name: str = None) -> None:
    # get absolute directory path
    log_dir_path = os.path.abspath(log_dir)
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    if not log_file_name:  # no log file name provided
        # file uses date/time to generate unique name
        datetime_obj = datetime.now()
        time_stamp = datetime_obj.strftime("%Y%d%m_%H%M%S")

        # generate file name:
        log_file_name = 'log_' + time_stamp + '.log'

    log_file_path = os.path.join(log_dir_path, log_file_name)

    # string representations of epoch information:
    # todo - rewards (and other params)
    agent_params = params_to_str(agent)
    agent_hyperparams = hyper_params_to_str(agent)
    agent_history = history_to_str(agent)

    # todo - generate strings above

    # appending is used to prevent accidental deletion
    with open(log_file_path, 'a+') as f:
        f.write('----------------------\n')
        f.write('EPOCH {}\n'.format(epoch))
        f.write('Total Steps: {}\n'.format(agent.step_count))
        f.write('----------------------\n')

        f.write('\nAGENT PARAMETERS\n')
        f.write(agent_params)

        f.write('\nAGENT HYPER-PARAMETERS\n')
        f.write(agent_hyperparams)

        f.write('\nAGENT HISTORY\n')
        f.write(agent_history)
