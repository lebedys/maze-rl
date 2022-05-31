import os
from datetime import datetime

import numpy as np

from src.agent.agent import Agent, Direction


def observation_to_str(observation: np.ndarray) -> str:
    # todo - parse and beautify
    return str(observation)


def params_to_str(agent: Agent) -> str:
    params_str = '''
    rewards: {rewards}
    '''.format(
        lr=agent.learning_rate,
        epsilon=agent.exploration_epsilon
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

    # todo - loop through path, observations, actions and append to

    for step in range(agent.step_count):
        row, col = agent.history['position'][step]
        observation_str = observation_to_str(agent.history['observation'][step])
        q_values = agent.history['q_values'][step]
        action = agent.history['action'][step].name
        is_random_choice = agent.history['is_random']
        is_finished = agent.history['is_finished']

        history_str += '''
        step = {step},
        position = ({row}, {col}),
        observation = {observation},
        q_values = {q_values},
        chosen action = {action},
        random exploration choice? = {is_random_choice},
        maze finished? = {is_finished}
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
    agent_params = ''  # params_to_str(agent)
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
        f.write(agent_hyperparams)

        f.write('\nAGENT HYPER-PARAMETERS\n')
        f.write(agent_hyperparams)

        f.write('\nAGENT HISTORY\n')
        f.write(agent_history)
