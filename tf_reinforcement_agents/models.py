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
                keras.layers.Conv2D(filters * 2, 3, kernel_initializer=initializer, use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.ELU()
            ]
            self._conv_block2 = [
                keras.layers.Conv2D(filters * 3, 3, kernel_initializer=initializer, use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.ELU()
            ]
            self._conv_block3 = [
                keras.layers.Conv2D(filters * 4, 3, kernel_initializer=initializer, use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.ELU()
            ]
            self._conv_block4 = [
                keras.layers.Conv2D(filters * 4, 3, kernel_initializer=initializer, use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.ELU()
            ]
            self._residual_block1 = [ResidualUnit(filters * 2, initializer, activation),
                                     ResidualUnit(filters * 2, initializer, activation),
                                     ResidualUnit(filters * 2, initializer, activation),
                                     ResidualUnit(filters * 2, initializer, activation)]
            self._residual_block2 = [ResidualUnit(filters * 3, initializer, activation),
                                     ResidualUnit(filters * 3, initializer, activation)]
            self._residual_block3 = [ResidualUnit(filters * 4, initializer, activation)]

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

    return ResidualModel()


def get_actor_critic3():
    import tensorflow as tf
    from tensorflow import keras

    K = keras.backend

    # from https://github.com/ageron/handson-ml2
    class MultiHeadAttention(keras.layers.Layer):
        def __init__(self, n_heads, causal=False, use_scale=False, **kwargs):
            self.n_heads = n_heads
            self.causal = causal
            self.use_scale = use_scale
            super().__init__(**kwargs)

        def build(self, batch_input_shape):
            self.dims = batch_input_shape[0][-1]
            self.q_dims, self.v_dims, self.k_dims = [self.dims // self.n_heads] * 3  # could be hyperparameters instead
            self.q_linear = keras.layers.Conv1D(self.n_heads * self.q_dims, kernel_size=1, use_bias=False)
            self.v_linear = keras.layers.Conv1D(self.n_heads * self.v_dims, kernel_size=1, use_bias=False)
            self.k_linear = keras.layers.Conv1D(self.n_heads * self.k_dims, kernel_size=1, use_bias=False)
            self.attention = keras.layers.Attention(causal=self.causal, use_scale=self.use_scale)
            self.out_linear = keras.layers.Conv1D(self.dims, kernel_size=1, use_bias=False)
            super().build(batch_input_shape)

        def _multi_head_linear(self, inputs, linear):
            shape = K.concatenate([K.shape(inputs)[:-1], [self.n_heads, -1]])
            outputs = linear(inputs)
            projected = K.reshape(outputs, shape)
            perm = K.permute_dimensions(projected, [0, 2, 1, 3])
            return K.reshape(perm, [shape[0] * self.n_heads, shape[1], -1])

        def call(self, inputs):
            q = inputs[0]
            v = inputs[1]
            k = inputs[2] if len(inputs) > 2 else v
            shape = K.shape(q)
            q_proj = self._multi_head_linear(q, self.q_linear)
            v_proj = self._multi_head_linear(v, self.v_linear)
            k_proj = self._multi_head_linear(k, self.k_linear)
            multi_attended = self.attention([q_proj, v_proj, k_proj])
            shape_attended = K.shape(multi_attended)
            reshaped_attended = K.reshape(multi_attended,
                                          [shape[0], self.n_heads, shape_attended[1], shape_attended[2]])
            perm = K.permute_dimensions(reshaped_attended, [0, 2, 1, 3])
            concat = K.reshape(perm, [shape[0], shape_attended[1], -1])
            return self.out_linear(concat)

    class AttentionModel(keras.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            initializer_random = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)

            self._dense0 = keras.layers.Dense(64, activation="relu")
            self._dense0_1 = keras.layers.Dense(64)
            # self._norm_0 = keras.layers.BatchNormalization()

            self._multi_attention1 = MultiHeadAttention(8, use_scale=True)
            self._norm1_1 = keras.layers.BatchNormalization()
            self._dense1_1 = keras.layers.Dense(64, activation="relu")
            self._dense1_2 = keras.layers.Dense(64)
            self._norm1_2 = keras.layers.BatchNormalization()

            # self._dense_food_1 = keras.layers.Dense(64, activation="relu")
            # self._dense_food_2 = keras.layers.Dense(64)
            # self._norm_food = keras.layers.BatchNormalization()
            # self._dense_scalars_1 = keras.layers.Dense(64, activation="relu")
            # self._dense_scalars_2 = keras.layers.Dense(64)
            # self._norm_scalars = keras.layers.BatchNormalization()

            self._multi_attention2 = MultiHeadAttention(8, use_scale=True)
            self._norm2_1 = keras.layers.BatchNormalization()
            self._dense2_1 = keras.layers.Dense(64, activation="relu")
            self._dense2_2 = keras.layers.Dense(64)
            self._norm2_2 = keras.layers.BatchNormalization()

            self._multi_attention3 = MultiHeadAttention(8, use_scale=True)
            self._norm3_1 = keras.layers.BatchNormalization()
            self._dense3_1 = keras.layers.Dense(64, activation="relu")
            self._dense3_2 = keras.layers.Dense(64)
            self._norm3_2 = keras.layers.BatchNormalization()

            # self._dense4_1 = keras.layers.Dense(128, activation="relu")

            self._logits = keras.layers.Dense(4, kernel_initializer=initializer_random)
            self._baseline = keras.layers.Dense(1, kernel_initializer=initializer_random,
                                                activation=keras.activations.tanh)

        def call(self, inputs, training=False):
            vectors, scalars_raw = inputs
            vectors = tf.cast(vectors, tf.float32)
            scalars_raw = tf.cast(scalars_raw, tf.float32)

            scalars_shape = tf.shape(scalars_raw)
            scalars_raw = tf.reshape(scalars_raw, [scalars_shape[0], 1, scalars_shape[-1]])
            scalars = tf.tile(scalars_raw, [1, 4, 1])
            geese = tf.concat([vectors, scalars], -1)

            # geese = self._norm_0(geese, training=training)

            # geese_vectors, food_vector = vectors[:, :-1, :], vectors[:, -1:, :]
            # geese_numbers = tf.constant([[[0, 0, 1],
            #                               [0, 1, 0],
            #                               [0, 1, 1],
            #                               [1, 0, 0]]], dtype=tf.float32)
            # geese_numbers = tf.tile(geese_numbers, [tf.shape(geese_vectors)[0], 1, 1])
            # geese_vectors = tf.concat([geese_vectors, geese_numbers], 2)
            # goose1 = geese_vectors
            # goose2 = tf.stack([geese_vectors[:, 1, :], geese_vectors[:, 2, :], geese_vectors[:, 3, :],
            #                    geese_vectors[:, 0, :], ], axis=1)
            # goose3 = tf.stack([geese_vectors[:, 2, :], geese_vectors[:, 3, :], geese_vectors[:, 0, :],
            #                    geese_vectors[:, 1, :], ], axis=1)
            # goose4 = tf.stack([geese_vectors[:, 3, :], geese_vectors[:, 0, :], geese_vectors[:, 1, :],
            #                    geese_vectors[:, 2, :], ], axis=1)

            # geese_raw = tf.stack([goose1, goose2, goose3, goose4], axis=1)
            # geese_raw = geese_vectors
            # geese_shape, food_shape = tf.shape(geese_raw), tf.shape(food_vector)
            # geese_shape = tf.shape(geese_raw)
            # food_raw = tf.reshape(food_vector, [food_shape[0] * food_shape[1], -1])

            geese_shape = tf.shape(geese)
            y = geese / 200  # 200 is a maximum number can be in observations

            y = tf.reshape(y, [geese_shape[0] * geese_shape[1], -1])
            y = self._dense0(y)
            y = self._dense0_1(y)

            y = tf.reshape(y, [geese_shape[0], geese_shape[1], -1])  # [batch, goose, parameters]
            x = self._multi_attention1([y, y])
            x = x + y
            x = tf.reshape(x, [geese_shape[0] * geese_shape[1], -1])
            x = self._norm1_1(x, training=training)
            y = self._dense1_1(x)
            y = self._dense1_2(y)
            x = x + y
            x = self._norm1_2(x, training=training)

            # geese = x
            # food = tf.tile(food, [4, 1])
            # geese = tf.concat([geese, food], 1)
            # x = self._dense_food_1(geese)
            # x = self._dense_food_2(x)
            # y = self._norm_food(x)

            # geese = x
            # scalars = tf.tile(scalars, [4, 1])
            # geese = tf.concat([geese, scalars], 1)
            # x = self._dense_scalars_1(geese)
            # x = self._dense_scalars_2(x)
            # y = self._norm_scalars(x, training=training)

            # y = tf.reshape(x, [geese_shape[0], geese_shape[1], -1])  # [batch, goose, parameters]
            # x = self._multi_attention2([y, y])
            # x = x + y
            # x = tf.reshape(x, [geese_shape[0] * geese_shape[1], -1])
            # x = self._norm2_1(x, training=training)
            # y = self._dense2_1(x)
            # y = self._dense2_2(y)
            # x = x + y
            # x = self._norm2_2(x, training=training)

            # y = tf.reshape(x, [geese_shape[0], geese_shape[1], -1])  # [batch, goose, parameters]
            # x = self._multi_attention3([y, y])
            # x = x + y
            # x = tf.reshape(x, [geese_shape[0] * geese_shape[1], -1])
            # x = self._norm3_1(x, training=training)
            # y = self._dense3_1(x)
            # y = self._dense3_2(y)
            # x = x + y
            # x = self._norm3_2(x, training=training)

            # x = tf.reshape(x, [geese_shape[0], geese_shape[1], -1])
            x = tf.reshape(x, [geese_shape[0], -1])

            # x = self._dense4_1(x)

            logits = self._logits(x)
            baseline = self._baseline(x)
            return logits, baseline

    class SimpleModel(keras.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            initializer_random = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)

            self._dense0_0 = keras.layers.Dense(100, kernel_regularizer=keras.regularizers.l2(0.01), activation="relu")
            self._dense0_0_1 = keras.layers.Dense(100, kernel_regularizer=keras.regularizers.l2(0.01))
            self._dense0_1 = keras.layers.Dense(100, kernel_regularizer=keras.regularizers.l2(0.01), activation="relu")
            self._dense0_1_1 = keras.layers.Dense(100, kernel_regularizer=keras.regularizers.l2(0.01))
            self._activation0 = keras.layers.Activation("relu")

            self._dense1 = keras.layers.Dense(500, kernel_regularizer=keras.regularizers.l2(0.01), activation="relu")
            self._dense2 = keras.layers.Dense(500, kernel_regularizer=keras.regularizers.l2(0.01), activation="relu")
            self._dense3 = keras.layers.Dense(500, kernel_regularizer=keras.regularizers.l2(0.01), activation="relu")

            self._logits = keras.layers.Dense(4, kernel_initializer=initializer_random)
            self._baseline = keras.layers.Dense(1, kernel_initializer=initializer_random,
                                                activation=keras.activations.tanh)

        def call(self, inputs, training=False):
            vectors, scalars_raw = inputs
            geese_raw = tf.cast(vectors, tf.float32)
            scalars_raw = tf.cast(scalars_raw, tf.float32)

            geese_shape = tf.shape(geese_raw)
            geese_raw = tf.reshape(geese_raw, [geese_shape[0] * geese_shape[1], -1])

            geese = self._dense0_0(geese_raw)
            geese = self._dense0_0_1(geese)
            scalars = self._dense0_1(scalars_raw)
            scalars = self._dense0_1_1(scalars)

            geese = tf.reshape(geese, [geese_shape[0], -1])

            x = tf.concat([geese, scalars], 1)
            x = self._activation0(x)

            x = self._dense1(x)
            x = self._dense2(x)
            x = self._dense3(x)

            logits = self._logits(x)
            baseline = self._baseline(x)
            return logits, baseline

    return AttentionModel()
