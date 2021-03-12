import random
from abc import ABC

import numpy as np
import tensorflow as tf

from goose_agent import models, misc
from goose_agent.abstract_agent import Agent


class RegularDQNAgent(Agent, ABC):

    def __init__(self, env_name, init_n_samples, *args, **kwargs):
        super().__init__(env_name, *args, **kwargs)

        # train a model from scratch
        if self._data is None:
            self._model = models.get_dqn(self._input_shape, self._n_outputs)
            # collect some data with a random policy (epsilon 1 corresponds to it) before training
            # self._collect_several_episodes(epsilon=1, n_episodes=self._sample_batch_size)
            self._collect_until_items_created(epsilon=self._epsilon, n_items=init_n_samples)
        # continue a model training
        elif self._data:
            self._model = models.get_dqn(self._input_shape, self._n_outputs)
            self._model.set_weights(self._data['weights'])
            # collect date with epsilon greedy policy
            self._collect_until_items_created(epsilon=self._epsilon, n_items=init_n_samples)

        reward = self._evaluate_episodes(num_episodes=100)
        print(f"Initial reward with a model policy is {reward:.2f}")

    def _epsilon_greedy_policy(self, obsns, epsilon, info):
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

    @tf.function
    def _training_step(self, actions, observations, rewards, dones, info):

        total_rewards, first_observations, last_observations, last_dones, last_discounted_gamma, second_actions = \
            self._prepare_td_arguments(actions, observations, rewards, dones)

        next_Q_values = self._model(last_observations)
        max_next_Q_values = tf.reduce_max(next_Q_values, axis=1)
        target_Q_values = total_rewards + (tf.constant(1.0) - last_dones) * last_discounted_gamma * max_next_Q_values
        target_Q_values = tf.expand_dims(target_Q_values, -1)
        mask = tf.one_hot(second_actions, self._n_outputs, dtype=tf.float32)
        with tf.GradientTape() as tape:
            all_Q_values = self._model(first_observations)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self._loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))


class CategoricalDQNAgent(Agent):

    def __init__(self, env_name, *args, **kwargs):
        super().__init__(env_name, *args, **kwargs)

        min_q_value = 0
        max_q_value = 1000
        self._n_atoms = 101
        self._support = tf.linspace(min_q_value, max_q_value, self._n_atoms)
        self._support = tf.cast(self._support, tf.float32)
        cat_n_outputs = self._n_outputs * self._n_atoms
        # train a model from scratch
        if self._data is None:
            self._model = models.get_dqn(self._input_shape, cat_n_outputs)
            # collect some data with a random policy (epsilon 1 corresponds to it) before training
            # self._collect_several_episodes(epsilon=1, n_episodes=self._sample_batch_size)
            self._collect_until_items_created(epsilon=1, n_items=self._sample_batch_size)
        # continue a model training
        elif self._data:
            self._model = models.get_dqn(self._input_shape, cat_n_outputs)
            self._model.set_weights(self._data['weights'])
            # collect date with epsilon greedy policy
            self._collect_several_episodes(epsilon=self._epsilon, n_episodes=self._sample_batch_size)

        reward = self._evaluate_episodes(num_episodes=100)
        print(f"Initial reward with a model policy is {reward}")

    def _epsilon_greedy_policy(self, obsns, epsilon, info):
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
                logits = self._predict(obs)
                logits = tf.reshape(logits, [-1, self._n_outputs, self._n_atoms])
                probabilities = tf.nn.softmax(logits)
                Q_values = tf.reduce_sum(self._support * probabilities, axis=-1)  # Q values expected return
                best_actions.append(np.argmax(Q_values[0]))
            return best_actions

    @tf.function
    def _training_step(self, actions, observations, rewards, dones, info):

        total_rewards, first_observations, last_observations, last_dones, last_discounted_gamma, second_actions = \
            self._prepare_td_arguments(actions, observations, rewards, dones)

        # Part 1: calculate new target (best) Q value distributions (next_best_probs)
        next_logits = self._model(last_observations)
        # reshape to (batch, n_actions, distribution support number of elements (atoms)
        next_logits = tf.reshape(next_logits, [-1, self._n_outputs, self._n_atoms])
        next_probabilities = tf.nn.softmax(next_logits)
        next_Q_values = tf.reduce_sum(self._support * next_probabilities, axis=-1)  # Q values expected return
        # get indices of max next Q values and get corresponding distributions
        max_args = tf.cast(tf.argmax(next_Q_values, 1), tf.int32)[:, None]
        batch_indices = tf.range(tf.cast(self._sample_batch_size, tf.int32))[:, None]
        next_qt_argmax = tf.concat([batch_indices, max_args], axis=-1)  # indices of the target Q value distributions
        next_best_probs = tf.gather_nd(next_probabilities, next_qt_argmax)

        # Part 2: calculate a new but non-aligned support of the target Q value distributions
        batch_support = tf.repeat(self._support[None, :], [self._sample_batch_size], axis=0)
        last_dones = tf.expand_dims(last_dones, -1)
        total_rewards = tf.expand_dims(total_rewards, -1)
        non_aligned_support = total_rewards + (tf.constant(1.0) - last_dones) * last_discounted_gamma * batch_support

        # Part 3: project the target Q value distributions to the basic (target_support) support
        target_distribution = misc.project_distribution(supports=non_aligned_support,
                                                        weights=next_best_probs,
                                                        target_support=self._support)

        # Part 4: Loss and update
        indices = tf.cast(batch_indices[:, 0], second_actions.dtype)
        reshaped_actions = tf.stack([indices, second_actions], axis=-1)
        with tf.GradientTape() as tape:
            logits = self._model(first_observations)
            logits = tf.reshape(logits, [-1, self._n_outputs, self._n_atoms])
            chosen_action_logits = tf.gather_nd(logits, reshaped_actions)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_distribution,
                                                           logits=chosen_action_logits)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
        return target_distribution, chosen_action_logits
