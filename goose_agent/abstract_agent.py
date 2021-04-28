import abc
import itertools as it

import numpy as np
import tensorflow as tf
import gym
import reverb


class Agent(abc.ABC):

    def __init__(self, env_name, config,
                 buffer_table_names, buffer_server_port,
                 data=None, make_checkpoint=False,
                 ):
        # environments; their hyperparameters
        self._train_env = gym.make(env_name)
        self._eval_env = gym.make(env_name)
        self._n_outputs = self._train_env.action_space.n  # number of actions
        space = self._train_env.observation_space
        self._n_players = len(space[0])  # number of players (geese)
        # env returns observations for all players, we need shape of any
        feature_maps_shape = space[0][0].shape  # height, width, channels
        scalar_features_shape = space[1].shape
        self._input_shape = (feature_maps_shape, scalar_features_shape)

        # data contains weighs, masks, and a corresponding reward
        self._data = data

        self._make_checkpoint = make_checkpoint

        # networks
        self._model = None
        self._target_model = None

        # hyperparameters for optimization
        self._optimizer = config["optimizer"]
        self._loss_fn = config["loss"]

        # buffer; hyperparameters for a reward calculation
        self._table_names = buffer_table_names
        # an object with a client, which is used to store data on a server
        self._replay_memory_client = reverb.Client(f'localhost:{buffer_server_port}')
        # make a batch size equal of a minimal size of a buffer
        self._sample_batch_size = config["batch_size"]
        self._n_steps = config["n_steps"]  # 1. amount of steps stored per item, it should be at least 2;
        # 2. for details see function _collect_trajectories_from_episode()
        self._discount_rate = config["discount_rate"]
        self._items_sampled = 0

        self._start_epsilon = None
        self._final_epsilon = None

    @tf.function
    def _predict(self, observation):
        return self._model(observation)

    @abc.abstractmethod
    def _policy(self, *args, **kwargs):
        raise NotImplementedError

    def _evaluate_episode(self, epsilon):
        """
        Epsilon 0 corresponds to greedy DQN _policy,
        if epsilon is None assume policy gradient _policy
        """
        obsns = self._eval_env.reset()
        obs_records = [(obsns[0][i], obsns[1]) for i in range(self._n_players)]
        rewards_storage = np.zeros(self._n_players)
        for step in it.count(0):
            if epsilon is None:
                actions, _ = self._policy(obs_records)
            else:
                actions = self._policy(obs_records, epsilon, info=None)
            obsns, rewards, dones, info = self._eval_env.step(actions)
            obs_records = [(obsns[0][i], obsns[1]) for i in range(self._n_players)]
            rewards_storage += np.asarray(rewards)
            if all(dones):
                break
        return rewards_storage.mean(), step

    def _evaluate_episodes(self, num_episodes=3, epsilon=None):
        episode_rewards = 0
        steps = 0
        for _ in range(num_episodes):
            reward, step = self._evaluate_episode(epsilon)
            episode_rewards += reward
            steps += step
        return episode_rewards / num_episodes, steps / num_episodes

    def _collect_trajectories_from_episode(self, epsilon):
        """
        Collects trajectories (items) to a buffer.
        A buffer contains items, each item consists of n_steps 'time steps';
        for a regular TD(0) update an item should have 2 time steps.
        One 'time step' contains (action, obs, reward, done);
        action, reward, done are for the current observation (or obs);
        e.g. action led to the obs, reward prior to the obs, if is it done at the current obs.

        this implementation creates writers for each player (goose) and stores
        n step trajectories for all of them

        if epsilon is None assume an off policy gradient method where policy_logits required
        """

        # initialize writers for all players
        writers = [self._replay_memory_client.writer(max_sequence_length=self._n_steps)
                   for _ in range(self._n_players)]
        obs_records = []
        info = None

        # todo: check scalars, they are similar for all geese
        obsns = self._train_env.reset()
        action, reward, done = tf.constant(-1), tf.constant(0.), tf.constant(0.)
        obsns = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.uint8), obsns)
        for i, writer in enumerate(writers):
            obs = obsns[0][i], obsns[1]
            obs_records.append(obs)
            if epsilon is None:
                policy_logits = tf.constant([0., 0., 0., 0.])
                writer.append((action, policy_logits, obs, reward, done))
            else:
                writer.append((action, obs, reward, done))

        for step in it.count(0):
            if epsilon is None:
                actions, policy_logits = self._policy(obs_records)
                policy_logits = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.float32),
                                                      policy_logits)
            else:
                actions = self._policy(obs_records, epsilon, info)
            obs_records = []
            # environment step receives actions and outputs observations for the dead players also
            # but it takes no effect
            obsns, rewards, dones, info = self._train_env.step(actions)
            actions = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.int32), actions)
            rewards = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.float32), rewards)
            dones = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.float32), dones)
            obsns = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.uint8), obsns)
            for i, writer in enumerate(writers):
                action, reward, done = actions[i], rewards[i], dones[i]
                obs = obsns[0][i], obsns[1]
                obs_records.append(obs)
                try:
                    if epsilon is None:
                        writer.append((action, policy_logits[i], obs, reward, done))
                    else:
                        writer.append((action, obs, reward, done))  # returns Runtime Error if a writer is closed
                    # if step >= start_itemizing:
                    for steps in range(2, self._n_steps + 1):
                        try:
                            writer.create_item(table=self._table_names[steps - 2], num_timesteps=steps, priority=1.)
                        except ValueError:
                            # stop new items creation if there are not enough buffered timesteps
                            break
                    if done:
                        writer.close()
                except RuntimeError:
                    # continue writing with a next writer if a current one is closed
                    continue
            if all(dones):
                break

    # def _collect_several_episodes(self, epsilon, n_episodes):
    #     for i in range(n_episodes):
    #         self._collect_trajectories_from_episode(epsilon)

    def _collect_until_items_created(self, n_items, epsilon=None):
        # collect more exp if we do not have enough for a batch
        # items are collected in several tables, where different tables save steps with different lengths
        # for example, if n_steps = 2, there is one table
        # if n_steps = 3, there are two tables with steps of lengths 2 and 3
        items_created = self._replay_memory_client.server_info()[self._table_names[0]][5].insert_stats.completed * \
                        (self._n_steps - 1)
        while items_created < n_items:
            self._collect_trajectories_from_episode(epsilon)
            items_created = self._replay_memory_client.server_info()[self._table_names[0]][5].insert_stats.completed * \
                            (self._n_steps - 1)

    def _prepare_td_arguments(self, actions, observations, rewards, dones, steps):
        exponents = tf.expand_dims(tf.range(steps - 1, dtype=tf.float32), axis=1)
        gammas = tf.fill([steps - 1, 1], self._discount_rate.numpy())
        discounted_gammas = tf.pow(gammas, exponents)

        total_rewards = tf.squeeze(tf.matmul(rewards[:, 1:], discounted_gammas))
        first_observations = tf.nest.map_structure(lambda x: x[:, 0, ...], observations)
        last_observations = tf.nest.map_structure(lambda x: x[:, -1, ...], observations)
        last_dones = dones[:, -1]
        last_discounted_gamma = self._discount_rate ** (steps - 1)
        second_actions = actions[:, 1]
        return total_rewards, first_observations, last_observations, last_dones, last_discounted_gamma, second_actions

    @abc.abstractmethod
    def _training_step(self, *args, **kwargs):
        raise NotImplementedError

    @tf.function
    def _train(self, samples_in):
        for i in range(self._n_steps - 1):
            action, obs, reward, done = samples_in[i].data
            key, probability, table_size, priority = samples_in[i].info
            experiences, info = (action, obs, reward, done), (key, probability, table_size, priority)
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

    def train_collect(self, iterations_number=20000, eval_interval=2000):

        target_model_update_interval = 3000
        epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self._start_epsilon,
            decay_steps=iterations_number,
            end_learning_rate=self._final_epsilon) if self._start_epsilon is not None else None

        weights = None
        mask = None
        rewards = 0
        steps = 0
        eval_counter = 0

        for step_counter in range(1, iterations_number + 1):
            # collecting
            items_created = self._replay_memory_client.server_info()[self._table_names[0]][5].insert_stats.completed * \
                            (self._n_steps - 1)
            # do not collect new experience if we have not used previous
            # train * X times more than collecting new experience
            if items_created * 20 < self._items_sampled:
                epsilon = epsilon_fn(step_counter) if self._start_epsilon is not None else None
                self._collect_trajectories_from_episode(epsilon)

            # dm-reverb returns tensors
            samples = [next(iterator) for iterator in self._iterators]
            # during training a batch of items is sampled n_steps - 1 times for all step sizes
            # e.g. 2 steps first, then 3 steps, etc.
            # todo: fix, it does not take into account skipping sometimes larger steps
            self._train(samples)
            self._items_sampled += self._sample_batch_size * (self._n_steps - 1)

            if step_counter % eval_interval == 0:
                eval_counter += 1
                mean_episode_reward, mean_steps = self._evaluate_episodes()
                epsilon = epsilon if self._start_epsilon is not None else 0
                print(f"Iteration:{step_counter:.2f}; "
                      f"Items sampled:{self._items_sampled:.2f}; "
                      f"Items created:{items_created:.2f}; "
                      f"Reward: {mean_episode_reward:.2f}; "
                      f"Steps: {mean_steps:.2f}; "
                      f"Epsilon: {epsilon:.2f}")
                rewards += mean_episode_reward
                steps += mean_steps

            # update target model weights
            if self._target_model and step_counter % target_model_update_interval == 0:
                weights = self._model.get_weights()
                self._target_model.set_weights(weights)

            # store weights at the last step
            if step_counter % iterations_number == 0:
                mean_episode_reward, mean_steps = self._evaluate_episodes(num_episodes=10)
                print(f"Final reward with a model policy is {mean_episode_reward:.2f}; "
                      f"Final average steps survived is {mean_steps:.2f}")
                output_reward = rewards / eval_counter
                output_steps = steps / eval_counter
                print(f"Average episode reward with a model policy is {output_reward:.2f}; "
                      f"Final average per episode steps survived is {output_steps:.2f}")

                weights = self._model.get_weights()
                mask = list(map(lambda x: np.where(np.abs(x) < 0.1, 0., 1.), weights))

                if self._make_checkpoint:
                    try:
                        checkpoint = self._replay_memory_client.checkpoint()
                    except RuntimeError as err:
                        print(err)
                        checkpoint = err
                else:
                    checkpoint = None

        return weights, mask, output_reward, output_steps, checkpoint
