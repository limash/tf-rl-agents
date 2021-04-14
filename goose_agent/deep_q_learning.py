import random
from abc import ABC

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from goose_agent import models, misc
from goose_agent.abstract_agent import Agent

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


class DQNAgent(Agent, ABC):

    def __init__(self, env_name, init_n_samples, *args, **kwargs):
        super().__init__(env_name, *args, **kwargs)

        # train a model from scratch
        if self._data is None:
            self._model = models.get_dqn(self._input_shape, self._n_outputs, is_duel=False)
        # continue a model training
        elif self._data:
            self._model = models.get_dqn(self._input_shape, self._n_outputs, is_duel=False)
            self._model.set_weights(self._data['weights'])

        self._target_model = models.get_dqn(self._input_shape, self._n_outputs, is_duel=False)
        self._target_model.set_weights(self._model.get_weights())

        self._collect_until_items_created(epsilon=self._epsilon, n_items=init_n_samples)

        reward, steps = self._evaluate_episodes(num_episodes=10)
        print(f"Initial reward with a model policy is {reward:.2f}, steps: {steps:.2f}")

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

    def _training_step(self, actions, observations, rewards, dones, steps, info):

        total_rewards, first_observations, last_observations, last_dones, last_discounted_gamma, second_actions = \
            self._prepare_td_arguments(actions, observations, rewards, dones, steps)

        next_Q_values = self._model(last_observations)
        best_next_actions = tf.argmax(next_Q_values, axis=1)
        next_mask = tf.one_hot(best_next_actions, self._n_outputs, dtype=tf.float32)
        next_best_Q_values = tf.reduce_sum((self._target_model(last_observations) * next_mask), axis=1)

        target_Q_values = total_rewards + (tf.constant(1.0) - last_dones) * last_discounted_gamma * next_best_Q_values
        target_Q_values = tf.expand_dims(target_Q_values, -1)
        mask = tf.one_hot(second_actions, self._n_outputs, dtype=tf.float32)
        with tf.GradientTape() as tape:
            all_Q_values = self._model(first_observations)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self._loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))


class RandomDQNAgent(DQNAgent):

    def _training_step(self, actions, observations, rewards, dones, steps, info):

        total_rewards, first_observations, last_observations, last_dones, last_discounted_gamma, second_actions = \
            self._prepare_td_arguments(actions, observations, rewards, dones, steps)

        next_Q_values = self._target_model(last_observations)
        next_best_Q_values = tfp.stats.percentile(next_Q_values, q=90., interpolation='linear', axis=1)

        # next_best_Q_values = tf.reduce_max(next_Q_values, axis=1)

        # idx_random = tf.random.uniform(shape=[], maxval=10, dtype=tf.int32)
        # if idx_random == 0:
        #     idx_random = tf.random.uniform(shape=[self._sample_batch_size], maxval=4, dtype=tf.int32)
        #     random_next_Q_values = misc.vector_slice(next_Q_values, idx_random)
        #     next_best_Q_values = random_next_Q_values
        # else:
        #     next_best_Q_values = tf.reduce_max(next_Q_values, axis=1)

        target_Q_values = total_rewards + (tf.constant(1.0) - last_dones) * last_discounted_gamma * next_best_Q_values
        target_Q_values = tf.expand_dims(target_Q_values, -1)
        mask = tf.one_hot(second_actions, self._n_outputs, dtype=tf.float32)
        with tf.GradientTape() as tape:
            all_Q_values = self._model(first_observations)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self._loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))


class CategoricalDQNAgent(Agent):

    def __init__(self, env_name, init_n_samples, *args, **kwargs):
        super().__init__(env_name, *args, **kwargs)

        min_q_value = -5
        max_q_value = 20
        self._n_atoms = 71
        self._support = tf.linspace(min_q_value, max_q_value, self._n_atoms)
        self._support = tf.cast(self._support, tf.float32)
        cat_n_outputs = self._n_outputs * self._n_atoms

        # train a model from scratch
        if self._data is None:
            self._model = models.get_dqn(self._input_shape, cat_n_outputs)
        # continue a model training
        elif self._data:
            self._model = models.get_dqn(self._input_shape, cat_n_outputs)
            self._model.set_weights(self._data['weights'])

        self._collect_until_items_created(epsilon=self._epsilon, n_items=init_n_samples)

        reward, steps = self._evaluate_episodes(num_episodes=10)
        print(f"Initial reward with a model policy is {reward:.2f}, steps: {steps:.2f}")

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

    def _training_step(self, actions, observations, rewards, dones, steps, info):

        total_rewards, first_observations, last_observations, last_dones, last_discounted_gamma, second_actions = \
            self._prepare_td_arguments(actions, observations, rewards, dones, steps)

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