import tensorflow as tf


CONF_DQN = {
    "agent": "regular",
    "iterations_number": 100000,
    "eval_interval": 5000,
    "batch_size": 64,
    "buffer_size": 500000,
    "n_steps": 2,  # 2 steps is a regular TD(0)
    "init_sample_epsilon": 1.,  # 1 means random sampling, for sampling before training
    "init_n_samples": 20000,
    "start_epsilon": 1.,  # start for polynomial decay eps schedule, it should be real (double)
    "final_epsilon": .1,
    "optimizer": tf.keras.optimizers.Adam(lr=1.e-3),
    "loss": tf.keras.losses.Huber(),
    "discount_rate": tf.constant(1., dtype=tf.float32)
}


CONF_RandomDQN = {
    "agent": "random",
    "iterations_number": 100000,
    "eval_interval": 5000,
    "batch_size": 64,
    "buffer_size": 500000,
    "n_steps": 2,  # 2 steps is a regular TD(0)
    "init_sample_epsilon": 1.,  # 1 means random sampling, for sampling before training
    "init_n_samples": 20000,
    "start_epsilon": 1.,  # start for polynomial decay eps schedule, it should be real (double)
    "final_epsilon": .1,
    "optimizer": tf.keras.optimizers.Adam(lr=1.e-3),
    "loss": tf.keras.losses.Huber(),
    "discount_rate": tf.constant(1., dtype=tf.float32)
}


CONF_CategoricalDQN = {
    "agent": "categorical",
    "iterations_number": 100000,
    "eval_interval": 5000,
    "batch_size": 64,
    "buffer_size": 500000,
    "n_steps": 2,  # 2 steps is a regular TD(0)
    "init_sample_epsilon": 1.,  # 1 means random sampling, for sampling before training
    "init_n_samples": 20000,
    "start_epsilon": 1.,  # start for polynomial decay eps schedule, it should be real (double)
    "final_epsilon": .1,
    "optimizer": tf.keras.optimizers.Adam(lr=1.e-4),
    # "optimizer": tf.keras.optimizers.RMSprop(lr=1.e-4, rho=0.95, momentum=0.0,
    #                                          epsilon=0.00001, centered=True),
    "loss": None,  # it is hard-coded in the categorical algorithm
    "discount_rate": tf.constant(.96, dtype=tf.float32)
}
