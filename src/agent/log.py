import os
from datetime import datetime

def log_agent(agent, epoch: int, log_dir: str = 'log/') -> None:
    # get absolute directory path
    log_dir_path = os.path.abspath(log_dir)
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    # file uses date/time to generate unique name
    datetime_obj = datetime.now()
    time_stamp = datetime_obj.strftime("%Y%d%m_%H%M%S")

    # generate file name:
    file_name = 'log_' + time_stamp + '.log'
    log_file_path = os.path.join(log_dir_path, file_name)

    # string representations of epoch information
    agent_params = ''
    agent_hyperparams = ''
    agent_history = ''

    # todo - generate strings above

    # appending is used to prevent accidental deletion
    with open(log_file_path, 'a+') as f:
        f.write('----------------------\n')
        f.write('EPOCH {}\n'.format(epoch))
        f.write('----------------------\n')

        f.write('\nAGENT PARAMETERS\n')
        f.write(agent_params)

        f.write('\nAGENT HYPER-PARAMETERS\n')
        f.write(agent_hyperparams)

        f.write('\nAGENT HISTORY\n')
        f.write(agent_history)
