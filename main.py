import pickle
from pathlib import Path

import ray
import reverb
import numpy as np

from goose_agent import deep_q_learning, policy_gradient, storage, misc
from config import *

config = CONF_ActorCritic
# config = CONF_DQN

AGENTS = {"DQN": deep_q_learning.DQNAgent,
          "percentileDQN": deep_q_learning.PercDQNAgent,
          "categoricalDQN": deep_q_learning.CategoricalDQNAgent,
          "actor-critic": policy_gradient.ACAgent}


def one_call(env_name, agent_name, data, checkpoint):
    if checkpoint is not None:
        path = str(Path(checkpoint).parent)  # due to https://github.com/deepmind/reverb/issues/12
        checkpointer = reverb.checkpointers.DefaultCheckpointer(path=path)
    else:
        checkpointer = None

    if config["buffer"] == "full_episode":
        # 1 table for an episode
        buffer = storage.UniformBuffer(num_tables=1,
                                       min_size=config["batch_size"], max_size=config["buffer_size"],
                                       checkpointer=checkpointer)
    else:
        # we need several tables for each step size
        buffer = storage.UniformBuffer(num_tables=config["n_points"]-1,
                                       min_size=config["batch_size"], max_size=config["buffer_size"],
                                       checkpointer=checkpointer)

    agent_object = AGENTS[agent_name]
    agent = agent_object(env_name, config,
                         buffer.table_names, buffer.server_port,
                         data, make_checkpoint=True)
    weights, mask, reward, steps, checkpoint = agent.train_collect(iterations_number=config["iterations_number"],
                                                                   eval_interval=config["eval_interval"])

    data = {
        'weights': weights,
        'mask': mask,
        'reward': reward
    }
    with open('data/data.pickle', 'wb') as f:
        pickle.dump(data, f, protocol=4)
    with open('data/checkpoint', 'w') as text_file:
        print(checkpoint, file=text_file)
    print("Done")


def multi_call(env_name, agent_name, data, checkpoint, plot=False):
    parallel_calls = 3
    ray.init(num_cpus=parallel_calls, num_gpus=1)

    if checkpoint is not None:
        path = str(Path(checkpoint).parent)  # due to https://github.com/deepmind/reverb/issues/12
        checkpointer = reverb.checkpointers.DefaultCheckpointer(path=path)
    else:
        checkpointer = None

    if config["buffer"] == "full_episode":
        # 1 table for an episode
        buffer = storage.UniformBuffer(num_tables=1,
                                       min_size=config["batch_size"], max_size=config["buffer_size"],
                                       checkpointer=checkpointer)
    else:
        # we need several tables for each step size
        buffer = storage.UniformBuffer(num_tables=config["n_points"]-1,
                                       min_size=config["batch_size"], max_size=config["buffer_size"],
                                       checkpointer=checkpointer)

    agent_object = AGENTS[agent_name]
    agent_object = ray.remote(num_gpus=1 / parallel_calls)(agent_object)
    agents = []
    for i in range(parallel_calls):
        make_checkpoint = True if i == 0 else False  # make a checkpoint only in the first worker
        agents.append(agent_object.remote(env_name, config,
                                          buffer.table_names, buffer.server_port,
                                          data, make_checkpoint))
    futures = [agent.train_collect.remote(iterations_number=config["iterations_number"],
                                          eval_interval=config["eval_interval"])
               for agent in agents]
    outputs = ray.get(futures)

    rewards_array = np.empty(parallel_calls)
    steps_array = np.empty(parallel_calls)
    weights_list, mask_list = [], []
    for count, (weights, mask, reward, steps, _) in enumerate(outputs):
        weights_list.append(weights)
        mask_list.append(mask)
        rewards_array[count] = reward
        steps_array[count] = steps
        print(f"Proc #{count}: Average reward = {reward:.2f}, Steps = {steps:.2f}")
        if plot:
            misc.plot_2d_array(weights[0], "Zero_lvl_with_reward_" + str(reward) + "_proc_" + str(count))
            misc.plot_2d_array(weights[2], "First_lvl_with_reward_" + str(reward) + "_proc_" + str(count))
    # argmax = rewards_array.argmax()
    argmax = steps_array.argmax()
    print(f"to save: Reward = {rewards_array[argmax]:.2f}, Steps = {steps_array[argmax]:.2f}")
    data = {
        'weights': weights_list[argmax],
        'mask': mask_list[argmax],
        'reward': rewards_array[argmax]
    }
    with open('data/data.pickle', 'wb') as f:
        pickle.dump(data, f, protocol=4)
    _, _, _, _, checkpoint = outputs[0]
    with open('data/checkpoint', 'w') as text_file:
        print(checkpoint, file=text_file)

    ray.shutdown()
    print("Done")


if __name__ == '__main__':
    try:
        with open('data/data.pickle', 'rb') as file:
            init_data = pickle.load(file)
    except FileNotFoundError:
        init_data = None

    try:
        init_checkpoint = open('data/checkpoint', 'r').read()
    except FileNotFoundError:
        init_checkpoint = None

    if config["multicall"]:
        multi_call(config["environment"], config["agent"], init_data, init_checkpoint)
    else:
        one_call(config["environment"], config["agent"], init_data, init_checkpoint)
