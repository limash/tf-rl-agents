import tensorflow as tf

from tf_reinforcement_agents.abstract_agent import Agent
from tf_reinforcement_agents import models

from tf_reinforcement_agents import storage, misc

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

        self._entropy_c = config["entropy_c"]
        self._entropy_c_decay = config["entropy_c_decay"]
        self._lambda = config["lambda"]

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
            self._training_step_full = tf.function(self._training_step_full)

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

        next_logits, baseline = self._model(last_observations)
        target_V = total_rewards + (tf.constant(1.0) - last_dones) * last_discounted_gamma * tf.squeeze(baseline)
        target_V = tf.expand_dims(target_V, -1)
        with tf.GradientTape() as tape:
            logits, V_values = self._model(first_observations)

            # critic loss
            critic_loss = self._loss_fn(target_V, V_values)
            # critic_loss_sum = .5 * tf.reduce_sum(tf.square(target_V - V_values))

            # actor loss
            # probs = tf.nn.softmax(logits)
            # mask = tf.one_hot(second_actions, self._n_outputs, dtype=tf.float32)
            # masked_probs = tf.reduce_sum(probs * mask, axis=1, keepdims=True)
            # logs = tf.math.log(masked_probs)
            # below is similar to above 4 lines, returns log probs from logits masked by actions
            logs = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                   labels=second_actions)
            # td error with truncated IS weights (rhos), it is a constant:
            with tape.stop_recording():
                # second logits correspond to second actions
                behaviour_logs = -tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=behaviour_policy_logits[:, 1, :],
                    labels=second_actions
                )
                log_rhos = logs - behaviour_logs
                rhos = tf.exp(log_rhos)
                clipped_rhos = tf.minimum(tf.constant(1.), rhos)
                clipped_rhos = tf.expand_dims(clipped_rhos, -1)
                td_error = clipped_rhos * (target_V - V_values)

            logs = tf.expand_dims(logs, -1)
            actor_loss = -1 * logs * td_error
            actor_loss = tf.reduce_mean(actor_loss)
            # actor_loss = tf.reduce_sum(actor_loss)

            # entropy loss
            entropy = misc.get_entropy(logits)
            entropy_loss = -1 * self._entropy_c * tf.reduce_mean(entropy)

            loss = actor_loss + critic_loss + entropy_loss
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

    def _training_step_full(self, actions, behaviour_policy_logits, observations, rewards, dones, steps, info):
        print("Tracing")

        actions = tf.transpose(actions)
        behaviour_policy_logits = tf.transpose(behaviour_policy_logits, perm=[1, 0, 2])
        maps = tf.transpose(observations[0], perm=[1, 0, 2, 3, 4])
        scalars = tf.transpose(observations[1], perm=[1, 0, 2])
        rewards = tf.transpose(rewards)
        dones = tf.transpose(dones)

        nsteps = tf.argmax(dones, axis=0, output_type=tf.int32)
        ta = tf.TensorArray(dtype=tf.float32, size=self._sample_batch_size, dynamic_size=False)
        for i in tf.range(self._sample_batch_size):
            row = tf.concat([tf.constant([0.]),
                             tf.linspace(0., 1., nsteps[i] + 1)[:-1],
                             tf.ones(steps - nsteps[i] - 1)], axis=0)
            ta = ta.write(i, row)
        progress = ta.stack()
        progress = tf.transpose(progress)

        # prepare a mask for the valid time steps
        # alive_positions = tf.where(actions != -1)
        # ones_array = tf.ones(alive_positions.shape[0])
        # mask = tf.scatter_nd(alive_positions, ones_array, actions.shape)
        mask2d = tf.where(actions == -1, 0., 1.)
        mask3d = tf.where(behaviour_policy_logits == 0., 0., 1.)

        # get final rewards, currently there is the only reward in the end of a game
        returns = rewards[-1, :]

        # behaviour_action_log_probs = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=behaviour_policy_logits,
        #                                                                              labels=actions)
        # it is almost similar to above line, but above probably won't work with cpus
        behaviour_action_log_probs = misc.get_prob_logs_from_logits(behaviour_policy_logits, actions,
                                                                    self._n_outputs)

        with tf.GradientTape() as tape:
            logits, values = tf.map_fn(self._model, (maps, scalars),
                                       fn_output_signature=[tf.TensorSpec((self._sample_batch_size,
                                                                           self._n_outputs), dtype=tf.float32),
                                                            tf.TensorSpec((self._sample_batch_size, 1),
                                                                          dtype=tf.float32)])
            logits = tf.roll(logits, shift=1, axis=0)  # shift by 1 along time dimension, to match a pattern
            values = tf.roll(values, shift=1, axis=0)  # where actions, logits, etc. are due to observation
            target_action_log_probs = misc.get_prob_logs_from_logits(logits, actions, self._n_outputs)

            with tape.stop_recording():
                log_rhos = target_action_log_probs - behaviour_action_log_probs
                rhos = tf.exp(log_rhos)
                # rhos_masked = tf.where(actions == -1, 0., rhos)  # use where to remove nans, should be outside tape
                rhos_masked = rhos * mask2d
                clipped_rhos = tf.minimum(tf.constant(1.), rhos_masked)

            # add final rewards to 'empty' spots in values
            values = tf.where(actions == -1, rewards, tf.squeeze(values))

            with tape.stop_recording():
                # calculate targets
                # td error with truncated IS weights (rhos), it is a constant:
                # targets = misc.prepare_td_lambda(tf.squeeze(values), returns, None, self._lambda, 1.)
                targets = misc.tf_prepare_td_lambda_no_rewards(tf.squeeze(values), returns, self._lambda, 1.)
                targets = targets * mask2d

            values = values * mask2d

            with tape.stop_recording():
                td_error = clipped_rhos * (targets - values)

            # critic loss
            # critic_loss = self._loss_fn(targets, values)
            critic_loss = .5 * tf.reduce_sum(tf.square(targets - values))

            # actor loss
            # use where to get rid of -infinities, but probably it causes inability to calculate grads
            # check https://stackoverflow.com/questions/33712178/tensorflow-nan-bug/42497444#42497444
            # target_action_log_probs = tf.where(actions == -1, 0., target_action_log_probs)
            target_action_log_probs = target_action_log_probs * mask2d
            actor_loss = -1 * target_action_log_probs * td_error
            # actor_loss = tf.reduce_mean(actor_loss)
            actor_loss = tf.reduce_sum(actor_loss)

            # entropy loss
            entropy = misc.get_entropy(logits, mask3d)
            # entropy_loss = -1 * self._entropy_c * tf.reduce_mean(entropy)
            entropy_loss = -1 * self._entropy_c * tf.reduce_sum(entropy * (1 - progress * (1 - self._entropy_c_decay)))

            loss = actor_loss + critic_loss + entropy_loss
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
