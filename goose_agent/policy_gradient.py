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

        if config["buffer"] == "n_points":
            self._datasets = [storage.initialize_dataset_with_logits(buffer_server_port,
                                                                     buffer_table_names[i],
                                                                     self._input_shape,
                                                                     self._sample_batch_size,
                                                                     i + 2) for i in range(self._n_points - 1)]
            self._iterators = [iter(self._datasets[i]) for i in range(self._n_points - 1)]
        elif config["buffer"] == "full_episode":
            dataset = storage.initialize_dataset_with_logits(buffer_server_port,
                                                             buffer_table_names[0],
                                                             self._input_shape,
                                                             self._sample_batch_size,
                                                             self._n_points,
                                                             is_episode=True)
            self._iterators = [iter(dataset), ]
        else:
            print("Check a buffer argument in config")
            raise LookupError

        # train a model from scratch
        if self._data is None:
            self._model = models.get_actor_critic(self._input_shape, self._n_outputs)
        # continue a model training
        else:
            self._model = models.get_actor_critic(self._input_shape, self._n_outputs)
            self._model.set_weights(self._data['weights'])

        if not config["debug"]:
            self._training_step = tf.function(self._training_step)

        self._collect_several_episodes(config["init_episodes"])

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

    def _training_step(self, actions, behaviour_policy_logits, observations, rewards, dones, steps, info):
        print("Tracing")
        total_rewards, first_observations, last_observations, last_dones, last_discounted_gamma, second_actions = \
            self._prepare_td_arguments(actions, observations, rewards, dones, steps)

        # maps = tf.transpose(observations[0], perm=[1, 0, 2, 3, 4])
        # scalars = tf.transpose(observations[1], perm=[1, 0, 2])
        # logits, values = tf.map_fn(self._model, (maps, scalars),
        #                            fn_output_signature=[tf.TensorSpec((self._sample_batch_size,
        #                                                                self._n_outputs), dtype=tf.float32),
        #                                                 tf.TensorSpec((self._sample_batch_size, 1),
        #                                                               dtype=tf.float32)])
        # actions = tf.transpose(actions)
        # target_action_log_probs = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
        #                                                                           labels=actions)
        # behaviour_policy_logits = tf.transpose(behaviour_policy_logits, perm=[1, 0, 2])
        # behaviour_action_log_probs = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=behaviour_policy_logits,
        #                                                                              labels=actions)
        # log_rhos = target_action_log_probs - behaviour_action_log_probs
        # rhos = tf.exp(log_rhos)
        # clipped_rhos = tf.minimum(tf.constant(1.), rhos)

        next_logits, baseline = self._model(last_observations)
        # baseline = values[-1]  # V values from the last observation
        target_V = total_rewards + (tf.constant(1.0) - last_dones) * last_discounted_gamma * tf.squeeze(baseline)
        # target_V = clipped_rhos * target_V
        target_V = tf.expand_dims(target_V, -1)
        with tf.GradientTape() as tape:
            logits, V_values = self._model(first_observations)

            critic_loss = tf.reduce_mean(self._loss_fn(target_V, V_values))

            # probs = tf.nn.softmax(logits)
            # mask = tf.one_hot(second_actions, self._n_outputs, dtype=tf.float32)
            # masked_probs = tf.reduce_sum(probs * mask, axis=1, keepdims=True)
            # logs = tf.math.log(masked_probs)
            # below is similar to above 4 lines
            logs = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                   labels=second_actions)
            # second logits correspond to second actions
            behaviour_logs = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=behaviour_policy_logits[:, 1, :],
                                                                             labels=second_actions)
            log_rhos = logs - behaviour_logs
            rhos = tf.exp(log_rhos)
            clipped_rhos = tf.minimum(tf.constant(1.), rhos)
            logs = tf.expand_dims(logs, -1)
            clipped_rhos = tf.expand_dims(clipped_rhos, -1)
            # to check: logs in actor loss should not be under stop_gradient, is it a case currently?
            td_error = tf.stop_gradient(clipped_rhos * (target_V - V_values))  # prevents updating critic part by actor
            actor_loss = -1 * logs * td_error
            actor_loss = tf.reduce_mean(actor_loss)

            loss = actor_loss + critic_loss
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
