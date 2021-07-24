# move all imports inside functions to use ray.remote multitasking

def mlp_layer(x):
    from tensorflow import keras
    import tensorflow.keras.layers as layers

    # initializer = "he_normal"
    # x = layers.Dense(512, kernel_initializer=initializer, activation='relu')(x)

    initializer = keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_in', distribution='truncated_normal')

    x = layers.Dense(1000, kernel_initializer=initializer,
                     kernel_regularizer=keras.regularizers.l2(0.01),
                     use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    # x = layers.ReLU()(x)

    x = layers.Dense(1000, kernel_initializer=initializer,
                     kernel_regularizer=keras.regularizers.l2(0.01),
                     use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    # x = layers.ReLU()(x)

    return x


def conv_layer(x):
    import tensorflow.keras.layers as layers
    from tensorflow import keras

    # x = layers.Conv2D(32, 8, kernel_initializer=initializer, strides=4, activation='relu')(x)
    # x = layers.Conv2D(64, 4, kernel_initializer=initializer, strides=2, activation='relu')(x)
    # x = layers.Conv2D(64, 3, kernel_initializer=initializer, strides=1, activation='relu')(x)

    initializer = keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_in', distribution='truncated_normal')

    x = layers.Conv2D(64, 5, kernel_initializer=initializer, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(64, 3, kernel_initializer=initializer, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(128, 3, kernel_initializer=initializer, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(128, 3, kernel_initializer=initializer, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)

    return x


def stem(input_shape, initializer=None):
    import tensorflow as tf
    from tensorflow import keras
    import tensorflow.keras.layers as layers

    feature_maps_shape, scalar_features_shape = input_shape
    # create inputs
    feature_maps_input = layers.Input(shape=feature_maps_shape, name="feature_maps", dtype=tf.uint8)
    scalar_feature_input = layers.Input(shape=scalar_features_shape, name="scalar_features", dtype=tf.uint8)
    inputs = [feature_maps_input, scalar_feature_input]
    # feature maps
    features_preprocessing_layer = keras.layers.Lambda(lambda obs: tf.cast(obs, tf.float32))
    features = features_preprocessing_layer(feature_maps_input)
    conv_output = conv_layer(features)
    # conv_output = handy_rl_resnet(features, initializer)
    # processing
    # h_head_filtered = keras.layers.Multiply()([tf.expand_dims(features[:, :, :, 0], -1), conv_output])
    # conv_proc_output = keras.layers.Conv2D(32, 1, kernel_initializer=initializer)(conv_output)
    flatten_conv_output = layers.Flatten()(conv_output)
    # x = layers.Dense(100, kernel_initializer=initializer,
    #                  kernel_regularizer=keras.regularizers.l2(0.01),
    #                  use_bias=False)(flatten_conv_output)
    # x = layers.BatchNormalization()(x)
    # # x = layers.ELU()(x)
    # x = layers.ReLU()(x)

    # concatenate inputs
    scalars_preprocessing_layer = keras.layers.Lambda(lambda obs: tf.cast(obs, tf.float32))
    scalars = scalars_preprocessing_layer(scalar_feature_input)
    # x = layers.Concatenate(axis=-1)([x, scalars])
    x = layers.Concatenate(axis=-1)([flatten_conv_output, scalars])
    # mlp
    x = mlp_layer(x)
    # x = layers.Dense(100, kernel_initializer=initializer,
    #                  kernel_regularizer=keras.regularizers.l2(0.01),
    #                  use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # # x = layers.ELU()(x)
    # x = layers.ReLU()(x)

    return inputs, x


def get_dqn(input_shape, n_outputs, is_duel=False):
    import tensorflow as tf
    from tensorflow import keras
    import tensorflow.keras.layers as layers

    inputs, x = stem(input_shape)
    # this initialization in the last layer decreases variance in the last layer
    initializer = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)
    # dueling
    if is_duel:
        state_values = layers.Dense(1, kernel_initializer=initializer)(x)
        raw_advantages = layers.Dense(n_outputs, kernel_initializer=initializer)(x)
        # advantages = raw_advantages - tf.reduce_max(raw_advantages, axis=1, keepdims=True)
        advantages = raw_advantages - tf.reduce_mean(raw_advantages, axis=1, keepdims=True)
        outputs = state_values + advantages
    else:
        outputs = layers.Dense(n_outputs, kernel_initializer=initializer)(x)

    model = keras.Model(inputs=[inputs], outputs=[outputs])

    return model


def get_actor_critic(input_shape, n_outputs):
    from tensorflow import keras
    import tensorflow.keras.layers as layers

    initializer_glorot = keras.initializers.GlorotUniform()
    initializer_random = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)
    bias_initializer = keras.initializers.Constant(-0.2)
    # initializer_vs = keras.initializers.VarianceScaling(
    #     scale=2.0, mode='fan_in', distribution='truncated_normal')

    inputs, x = stem(input_shape, initializer_glorot)

    policy_logits = layers.Dense(n_outputs,
                                 kernel_initializer=initializer_glorot)(x)  # are not normalized logs
    # baseline = layers.Dense(1, kernel_initializer=initializer_random, bias_initializer=bias_initializer,
    #                         activation=keras.activations.tanh)(x)
    # baseline = layers.Dense(1, kernel_initializer=initializer_random, bias_initializer=bias_initializer)(x)
    baseline = layers.Dense(1, kernel_initializer=initializer_random)(x)

    model = keras.Model(inputs=[inputs], outputs=[policy_logits, baseline])

    return model


def get_actor_critic2():
    import tensorflow as tf
    from tensorflow import keras

    # physical_devices = tf.config.list_physical_devices('GPU')
    # if len(physical_devices) > 0:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def circular_padding(x):
        x = tf.concat([x[:, -1:, :, :], x, x[:, :1, :, :]], 1)
        x = tf.concat([x[:, :, -1:, :], x, x[:, :, :1, :]], 2)
        return x

    class ResidualUnit(keras.layers.Layer):
        def __init__(self, filters, initializer, activation, **kwargs):
            super().__init__(**kwargs)

            self._filters = filters
            self._activation = activation
            self._main_layers = [
                keras.layers.Conv2D(filters, 3, kernel_initializer=initializer),  # , use_bias=False),
                keras.layers.BatchNormalization()
            ]

        def call(self, inputs, **kwargs):
            x = inputs
            x = circular_padding(x)

            for layer in self._main_layers:
                x = layer(x)
            return self._activation(inputs + x)

        def compute_output_shape(self, batch_input_shape):
            batch, x, y, _ = batch_input_shape
            return [batch, x, y, self._filters]

    class SmallResidualModel(keras.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            filters = 32
            layers = 12

            # initializer = keras.initializers.HeNormal
            initializer = keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
            initializer_random = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)
            activation = keras.activations.relu

            self._conv_block_first = [
                keras.layers.Conv2D(filters, 3, kernel_initializer=initializer),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU()
            ]
            # self._conv_block_last = [
            #     keras.layers.Conv2D(filters, 3, kernel_initializer=initializer, padding='same'),
            #     keras.layers.BatchNormalization(),
            #     keras.layers.ReLU()
            # ]
            self._residual_block = [ResidualUnit(filters, initializer, activation) for _ in range(layers)]
            # self._residual_block = [ResidualUnit(filters, initializer, activation),
            #                         ResidualUnit(filters, initializer, activation),
            #                         ResidualUnit(filters, initializer, activation)]

            self._logits = keras.layers.Dense(4, kernel_initializer=initializer_random)
            self._baseline = keras.layers.Dense(1, kernel_initializer=initializer_random,
                                                activation=keras.activations.tanh)

        def call(self, inputs, training=None, mask=None):
            maps, scalars = inputs
            maps = tf.cast(maps, tf.float32)
            # scalars = tf.cast(scalars, tf.float32)

            x = maps

            x = circular_padding(x)
            for layer in self._conv_block_first:
                x = layer(x)

            for layer in self._residual_block:  # + self._conv_block_last:
                x = layer(x)

            shape_x = tf.shape(x)
            y = tf.reshape(x, (shape_x[0], -1, shape_x[-1]))
            y = tf.reduce_mean(y, axis=1)

            z = (x * maps[:, :, :, :1])
            shape_z = tf.shape(z)
            z = tf.reshape(z, (shape_z[0], -1, shape_z[-1]))
            z = tf.reduce_sum(z, axis=1)

            baseline = self._baseline(tf.concat([y, z], axis=1))
            policy_logits = self._logits(z)

            return policy_logits, baseline

        def get_config(self):
            pass

    class ResidualModel(keras.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            filters = 32

            initializer = keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
            initializer_random = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)
            activation = tf.nn.elu

            self._conv_block1 = [
                keras.layers.Conv2D(filters*2, 3, kernel_initializer=initializer, use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.ELU()
            ]
            self._conv_block2 = [
                keras.layers.Conv2D(filters*3, 3, kernel_initializer=initializer, use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.ELU()
            ]
            self._conv_block3 = [
                keras.layers.Conv2D(filters*4, 3, kernel_initializer=initializer, use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.ELU()
            ]
            self._conv_block4 = [
                keras.layers.Conv2D(filters*4, 3, kernel_initializer=initializer, use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.ELU()
            ]
            self._residual_block1 = [ResidualUnit(filters*2, initializer, activation),
                                     ResidualUnit(filters*2, initializer, activation),
                                     ResidualUnit(filters*2, initializer, activation),
                                     ResidualUnit(filters*2, initializer, activation)]
            self._residual_block2 = [ResidualUnit(filters*3, initializer, activation),
                                     ResidualUnit(filters*3, initializer, activation)]
            self._residual_block3 = [ResidualUnit(filters*4, initializer, activation)]

            self._flatten = keras.layers.Flatten()
            self._dense_block = [
                keras.layers.Dense(500, kernel_initializer=initializer,
                                   kernel_regularizer=keras.regularizers.l2(0.01),
                                   use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.ELU(),
                keras.layers.Dense(500, kernel_initializer=initializer,
                                   kernel_regularizer=keras.regularizers.l2(0.01),
                                   use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.ELU()
            ]
            self._baseline_block = [
                keras.layers.Dense(100, kernel_initializer=initializer,
                                   kernel_regularizer=keras.regularizers.l2(0.01),
                                   use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.ELU(),
                keras.layers.Dense(1, kernel_initializer=initializer_random)
            ]
            self._logits_block = [
                keras.layers.Dense(100, kernel_initializer=initializer,
                                   kernel_regularizer=keras.regularizers.l2(0.01),
                                   use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.ELU(),
                keras.layers.Dense(4, kernel_initializer=initializer_random)
            ]

        def call(self, inputs, training=None, mask=None):
            maps, scalars = inputs
            maps = tf.cast(maps, tf.float32)
            scalars = tf.cast(scalars, tf.float32)

            x = maps

            x = circular_padding(x)
            for layer in self._conv_block1 + self._residual_block1:
                x = layer(x)
            for layer in self._conv_block2 + self._residual_block2:
                x = layer(x)
            for layer in self._conv_block3 + self._residual_block3 + self._conv_block4:
                x = layer(x)

            x = self._flatten(x)

            y = tf.concat([x, scalars], axis=-1)
            for layer in self._dense_block:
                y = layer(y)

            baseline = y
            for layer in self._baseline_block:
                baseline = layer(baseline)

            logits = y
            for layer in self._logits_block:
                logits = layer(logits)

            return logits, baseline

        def get_config(self):
            pass

    return SmallResidualModel()
