import tensorflow as tf


CONF_1 = {
    "agent": "regular",
    "iterations_number": 20000,
    "batch_size": 64,
    "buffer_size": 500000,
    "n_steps": 2,  # 2 steps is a regular TD(0)
    "init_sample_epsilon": .1,  # 1 means random sampling, for sampling before training
    "init_n_samples": 1000,
    "start_epsilon": .1,  # start for polynomial decay eps schedule, it should be real (double)
    "final_epsilon": .1,
    "optimizer": tf.keras.optimizers.Adam(lr=1.e-5),
    "loss": tf.keras.losses.Huber(),
    "discount_rate": tf.constant(1., dtype=tf.float32)
}
