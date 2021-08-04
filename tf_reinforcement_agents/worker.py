import random
import pickle
# import time
import time
from abc import ABC

import tensorflow as tf
import numpy as np
import ray
from ray.util.queue import Empty

from tf_reinforcement_agents.abstract_agent import Agent
from tf_reinforcement_agents import models

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Collector(Agent, ABC):

    def __init__(self, env_name, config,
                 buffer_table_names, buffer_server_port,
                 *args, **kwargs):
        super().__init__(env_name, config,
                         buffer_table_names, buffer_server_port,
                         *args, **kwargs)

        if self._is_policy_gradient:
            # self._model = models.get_actor_critic(self._input_shape, self._n_outputs)
            self._model = models.get_actor_critic2()
            self._policy = self._pg_policy
        else:
            self._model = models.get_dqn(self._input_shape, self._n_outputs, is_duel=False)
            self._policy = self._dqn_policy

        dummy_input = (tf.ones(self._input_shape[0], dtype=tf.uint8),
                       tf.ones(self._input_shape[1], dtype=tf.uint8))
        dummy_input = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), dummy_input)
        self._predict(dummy_input)
        if self._data is not None:
            self._model.set_weights(self._data['weights'])

    def _dqn_policy(self, obsns, epsilon, info):
        if np.random.rand() < epsilon:
            # the first step after reset is arbitrary
            if info is None:
                available_actions = [0, 1, 2, 3]
                actions = [random.choice(available_actions) for _ in range(self._n_players)]
            # other random actions are within actions != opposite to the previous ones
            else:
                actions = [random.choice(info[i]['allowed_actions']) for i in range(self._n_players)]
            return actions
        else:
            # it receives observations for all geese and predicts best actions one by one
            best_actions = []
            obsns = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), obsns)
            for i in range(self._n_players):
                obs = obsns[i]
                Q_values = self._predict(obs)
                best_actions.append(np.argmax(Q_values[0]))
            return best_actions

    def _pg_policy(self, obsns, is_random=False):
        actions = []
        logits = []
        obsns = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), obsns)
        for i in range(self._n_players):
            obs = obsns[i]
            if is_random:
                policy_logits = tf.zeros([tf.shape(obs[0])[0], self._n_outputs])
            else:
                policy_logits, _ = self._predict(obs)
            action = tf.random.categorical(policy_logits, num_samples=1, dtype=tf.int32)
            actions.append(action.numpy()[0][0])
            logits.append(policy_logits.numpy()[0])
            # probabilities = tf.nn.softmax(policy_logits)
            # return np.argmax(probabilities[0])
        return actions, logits

    def do_collect(self):
        num_collects = 0
        num_updates = 0

        while True:
            # trainer will switch to done on the last iteration
            is_done = ray.get(self._workers_info.get_done.remote())
            if is_done:
                # print("Collecting is done.")
                return num_collects, num_updates
            # get the current turn, so collectors (workers) update weights one by one
            curr_worker = ray.get(self._workers_info.get_global_v.remote())
            # check the current turn
            if curr_worker == self._worker_id:
                if not self._ray_queue.empty():  # see below
                    try:
                        # block = False will cause an exception if there is no data in the queue,
                        # which is not handled by a ray queue (incompatibility with python 3.8 ?)
                        weights = self._ray_queue.get(block=False)
                        if curr_worker == self._num_collectors:
                            # print(f"Worker {curr_worker} updates weights")
                            ray.get(self._workers_info.set_global_v.remote(1))
                            num_updates += 1
                        elif curr_worker < self._num_collectors:
                            ray.get(self._workers_info.set_global_v.remote(curr_worker + 1))
                            # print(f"Worker {curr_worker} update weights")
                            num_updates += 1
                        else:
                            print("Wrong worker")
                            raise NotImplementedError
                    except Empty:
                        weights = None
                else:
                    weights = None
            else:
                weights = None

            if weights is not None:
                self._model.set_weights(weights)
                # print("Weights are updated")

            epsilon = None
            # t1 = time.time()
            num_collects += 1
            if num_collects < 1000 or num_collects % 100 == 0:
                self._collect(epsilon, is_random=True)
                # print("Episode with a random trajectory was collected; "
                #       f"Num of collects: {num_collects}")
            else:
                self._collect(epsilon)
                # print(f"Num of collects: {num_collects}")
            # t2 = time.time()
            # print(f"Collecting. Time: {t2 - t1}")


class Evaluator(Agent, ABC):

    def __init__(self, env_name, config,
                 buffer_table_names, buffer_server_port,
                 *args, **kwargs):
        super().__init__(env_name, config,
                         buffer_table_names, buffer_server_port,
                         *args, **kwargs)

        if self._is_policy_gradient:
            # self._model = models.get_actor_critic(self._input_shape, self._n_outputs)
            self._model = models.get_actor_critic2()
            self._eval_model = models.get_actor_critic2()
            # self._policy = self._pg_policy
        else:
            self._model = models.get_dqn(self._input_shape, self._n_outputs, is_duel=False)
            # self._policy = self._dqn_policy

        dummy_input = (tf.ones(self._input_shape[0], dtype=tf.uint8),
                       tf.ones(self._input_shape[1], dtype=tf.uint8))
        dummy_input = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), dummy_input)
        self._predict(dummy_input)
        self._eval_predict(dummy_input)

        with open('data/eval/data.pickle', 'rb') as file:
            data = pickle.load(file)
        self._eval_model.set_weights(data['weights'])

    @tf.function
    def _eval_predict(self, observation):
        return self._eval_model(observation)

    def evaluate_episode(self):
        obs_records = self._eval_env.reset()
        obsns = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), obs_records)
        rewards_storage = np.zeros(self._n_players)
        while True:
            actions = []
            for i in range(self._n_players):
                # policy_logits, _ = self._predict(obsns[i]) if i < 2 else self._eval_predict(obsns[i])
                policy_logits, _ = self._eval_predict(obsns[i]) if i < 2 else self._predict(obsns[i])
                action = tf.random.categorical(policy_logits, num_samples=1, dtype=tf.int32)
                actions.append(action.numpy()[0][0])

            obs_records, rewards, dones, info = self._eval_env.step(actions)
            rewards_storage += np.asarray(rewards)
            if all(dones):
                break
        winner = rewards_storage.argmax()
        return rewards_storage, winner

    def evaluate_episodes(self):
        wins = 0
        total_rewards = np.zeros(self._n_players)
        for _ in range(100):
            rewards, winner = self.evaluate_episode()
            # if winner == 0 or winner == 1:
            if winner == 2 or winner == 3:
                wins += 1
            total_rewards += rewards
        return total_rewards, wins

    def do_evaluate(self):
        while True:
            is_done = ray.get(self._workers_info.get_done.remote())
            if is_done:
                # print("Evaluation is done.")
                time.sleep(1)  # is needed to have time to print the last 'total wins'
                return 'Done'
            while True:
                weights, step = ray.get(self._workers_info.get_current_weights.remote())
                if weights is None:
                    time.sleep(1)
                else:
                    self._model.set_weights(weights)
                    break

            total_rewards, wins = self.evaluate_episodes()
            print(f"Total rewards: {total_rewards}; Total wins: {wins}; Time step: {step}")
