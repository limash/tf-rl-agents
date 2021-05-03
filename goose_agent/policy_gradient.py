import numpy as np
import tensorflow as tf

from goose_agent.abstract_agent import Agent
from goose_agent import models

from goose_agent import storage

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


class ACAgent(Agent):

    def __init__(self, env_name, config,
                 buffer_table_names, buffer_server_port,
                 *args, **kwargs):
        super().__init__(env_name, config,
                         buffer_table_names, buffer_server_port,
                         *args, **kwargs)

        self._datasets = [storage.initialize_dataset_with_logits(buffer_server_port,
                                                                 buffer_table_names[i],
                                                                 self._input_shape,
                                                                 self._sample_batch_size,
                                                                 i + 2) for i in range(self._n_points - 1)]
        self._iterators = [iter(self._datasets[i]) for i in range(self._n_points - 1)]

        # train a model from scratch
        if not self._data:
            self._model = models.get_actor_critic(self._input_shape, self._n_outputs)
        # continue a model training
        else:
            self._model = models.get_actor_critic(self._input_shape, self._n_outputs)
            self._model.set_weights(self._data['weights'])

        self._collect_until_items_created(n_items=config["init_n_samples"])

        reward, steps = self._evaluate_episodes(num_episodes=10)
        print(f"Initial reward with a model policy is {reward:.2f}, steps: {steps:.2f}")

    def _policy(self, obsns):
        actions = []
        logits = []
        obsns = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), obsns)
        for i in range(self._n_players):
            obs = obsns[i]
            policy_logits, _ = self._predict(obs)
            action = tf.random.categorical(policy_logits, num_samples=1, dtype=tf.int32)
            actions.append(action.numpy()[0][0])
            logits.append(policy_logits.numpy()[0])
            # probabilities = tf.nn.softmax(policy_logits)
            # return np.argmax(probabilities[0])

        return actions, logits

    # @tf.function
    def _train(self, samples_in):
        for i in range(self._n_points - 1):
            action, policy_logits, obs, reward, done = samples_in[i].data
            key, probability, table_size, priority = samples_in[i].info
            experiences, info = (action, policy_logits, obs, reward, done), (key, probability, table_size, priority)
            # self._training_step(*experiences, steps=i + 2, info=info)
            #
            trigger = tf.random.uniform(shape=[])
            # if i > 1 and trigger > 1 / i:
            # if i > 0 and trigger > max(1. / (i + 1.), 0.25):
            if i > 0 and trigger > 1. / (i + 1.):
                pass
            else:
                self._training_step(*experiences, steps=i + 2, info=info)
            # if i == 0 or i == self._n_steps - 2:
            #     self._training_step(*experiences, steps=i + 2, info=info)

    def _training_step(self, actions, policy_logits, observations, rewards, dones, steps, info):

        total_rewards, first_observations, last_observations, last_dones, last_discounted_gamma, second_actions = \
            self._prepare_td_arguments(actions, observations, rewards, dones, steps)

        next_logits, baseline = self._model(last_observations)
        max_next_Q_values = tf.reduce_max(next_Q_values, axis=1)
        target_Q_values = total_rewards + (tf.constant(1.0) - last_dones) * last_discounted_gamma * max_next_Q_values
        target_Q_values = tf.expand_dims(target_Q_values, -1)
        mask = tf.one_hot(second_actions, self._n_outputs, dtype=tf.float32)
        with tf.GradientTape() as tape:
            all_logits, all_Q_values = self._model(first_observations)
            probs = tf.nn.softmax(all_logits)
            masked_probs = tf.reduce_sum(probs * mask, axis=1, keepdims=True)
            logs = tf.math.log(masked_probs)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            td_error = tf.stop_gradient(target_Q_values - Q_values)  # to prevent updating critic part by actor
            actor_loss = -1*logs*td_error
            actor_loss = tf.reduce_mean(actor_loss)
            critic_loss = tf.reduce_mean(self._loss_fn(target_Q_values, Q_values))
            loss = actor_loss + critic_loss
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
