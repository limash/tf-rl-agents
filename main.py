import pickle
from pathlib import Path

import ray
import reverb
import numpy as np

from goose_agent import deep_q_learning, storage, misc

AGENTS = {"regular": deep_q_learning.RegularDQNAgent,
          "categorical": deep_q_learning.CategoricalDQNAgent}

BUFFERS = {"regular": storage.UniformBuffer,
           "categorical": storage.UniformBuffer}

BATCH_SIZE = 32
BUFFER_SIZE = 500000
N_STEPS = 2  # 2 steps is a regular TD(0)

INIT_SAMPLE_EPS = 1.  # 1 means random sampling, for sampling before training
INIT_N_SAMPLES = 20000

EPS = 1.  # start for polynomial decay eps schedule, it should be real (double)


def one_call(env_name, agent_name, data, checkpoint):
    if checkpoint is not None:
        path = str(Path(checkpoint).parent)  # due to https://github.com/deepmind/reverb/issues/12
        checkpointer = reverb.checkpointers.DefaultCheckpointer(path=path)
    else:
        checkpointer = None
    buffer = BUFFERS[agent_name](min_size=BATCH_SIZE, max_size=BUFFER_SIZE, checkpointer=checkpointer)

    agent_object = AGENTS[agent_name]
    agent = agent_object(env_name, INIT_N_SAMPLES,
                         buffer.table_name, buffer.server_port, buffer.min_size,
                         N_STEPS, INIT_SAMPLE_EPS,
                         data, make_checkpoint=True)
    weights, mask, reward, checkpoint = agent.train_collect(iterations_number=100000, epsilon=EPS)

    data = {
        'weights': weights,
        'mask': mask,
        'reward': reward
    }
    with open('data/data.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    with open('data/checkpoint', 'w') as text_file:
        print(checkpoint, file=text_file)
    print("Done")


if __name__ == '__main__':
    goose = 'gym_goose:goose-full_control-v0'

    try:
        with open('data/data.pickle', 'rb') as file:
            init_data = pickle.load(file)
    except FileNotFoundError:
        init_data = None

    try:
        init_checkpoint = open('data/checkpoint', 'r').read()
    except FileNotFoundError:
        init_checkpoint = None

    one_call(goose, 'regular', init_data, init_checkpoint)
