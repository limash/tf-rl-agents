# move all imports inside functions to use ray.remote multitasking

def mlp_layer(x):
    from tensorflow import keras
    import tensorflow.keras.layers as layers

    x = layers.Dense(1000, kernel_initializer="he_normal",
                     kernel_regularizer=keras.regularizers.l2(0.01),
                     use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)

    x = layers.Dense(500, kernel_initializer="he_normal",
                     kernel_regularizer=keras.regularizers.l2(0.01),
                     use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)

    return x


def conv_layer(x):
    import tensorflow.keras.layers as layers

    # x is (7, 11, 4)
    x = layers.Conv2D(64, 5, activation='elu', padding='same')(x)
    x = layers.Conv2D(64, 3, activation='elu', padding='valid')(x)
    x = layers.Conv2D(128, 3, activation='elu', padding='valid')(x)
    x = layers.Conv2D(128, 3, activation='elu', padding='valid')(x)

    return x


def get_dqn(input_shape, n_outputs):
    from tensorflow import keras
    import tensorflow.keras.layers as layers

    feature_maps_shape, scalar_features_shape = input_shape
    # create inputs
    feature_maps_input = layers.Input(shape=feature_maps_shape, name="feature_maps")
    scalar_feature_input = layers.Input(shape=scalar_features_shape, name="scalar_features")
    inputs = [feature_maps_input, scalar_feature_input]
    # feature maps
    conv_output = conv_layer(feature_maps_input)
    flatten_conv_output = layers.Flatten()(conv_output)
    # concatenate inputs
    x = layers.Concatenate(axis=-1)([flatten_conv_output, scalar_feature_input])
    # mlp
    x = mlp_layer(x)
    outputs = layers.Dense(n_outputs)(x)

    model = keras.Model(inputs=[inputs], outputs=[outputs])

    return model


def get_actor_critic(input_shape, n_outputs):
    from tensorflow import keras
    import tensorflow.keras.layers as layers

    inputs = layers.Input(shape=input_shape)
    x = mlp_layer(inputs)
    x = layers.Activation("relu")(x)
    logits = layers.Dense(n_outputs)(x)  # are not normalized logs
    q_values = layers.Dense(n_outputs)(x)
    model = keras.Model(inputs=[inputs], outputs=[logits, q_values])
    return model


def get_sparse(weights_in, mask_in):
    from abc import ABC

    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    class SparseSublayer(keras.layers.Layer):
        def __init__(self, w_init):
            super(SparseSublayer, self).__init__()
            self._w = tf.Variable(initial_value=w_init, trainable=True, dtype=tf.float32)

        def call(self, inputs, **kwargs):
            return tf.matmul(inputs, self._w)

    class SparseLayer(keras.layers.Layer):
        def __init__(self, w_init, b_init, mask):
            super(SparseLayer, self).__init__()
            # w size is (input_dimensions, units)

            bool_mask = mask.astype(np.bool)
            self._w = []
            self._mask = []
            self._num_connections = []
            num_neurons = self._num_neurons = w_init.shape[-1]
            for i in range(num_neurons):
                weights = w_init[:, i]
                masked_weights = weights[bool_mask[:, i]]
                # making a column vector, it is necessary for tf matrix multiplication
                masked_weights_column = masked_weights[..., None]
                # self._w.append(tf.Variable(initial_value=masked_weights_column, trainable=True, dtype=tf.float32))
                self._w.append(SparseSublayer(masked_weights_column))
                self._mask.append(tf.constant(bool_mask[:, i], dtype=tf.bool))
                self._num_connections.append(tf.constant(np.sum(mask[:, i]).astype(np.int32), dtype=tf.int32))

            self._b = tf.Variable(initial_value=b_init, trainable=True, dtype=tf.float32)

        def call(self, inputs, **kwargs):
            neurons = []
            for i in range(self._num_neurons):
                # reshape mask to (batch_size x inputs_size)
                mask = tf.broadcast_to(self._mask[i], [inputs.shape[0], self._mask[i].shape[0]])
                # mask inputs
                masked_inputs = tf.boolean_mask(inputs, mask)
                # restore dimensions after masking
                # reshaped_masked_inputs = tf.reshape(
                #     masked_inputs, [inputs.shape[0], tf.reduce_sum(tf.cast(self._mask[i], tf.int32)).numpy()])
                reshaped_masked_inputs = tf.reshape(masked_inputs, [inputs.shape[0], self._num_connections[i]])
                # matrix multiplication for one neuron
                # neuron = tf.matmul(reshaped_masked_inputs, self._w[i]) + self._b[i]
                neuron = self._w[i](reshaped_masked_inputs) + self._b[i]
                neurons.append(neuron)

            result = tf.stack([*neurons], axis=1)
            result = result[..., 0]  # all except the last dimension
            return result

    class SparseMLP(keras.Model, ABC):
        def __init__(self, weights, mask):
            super(SparseMLP, self).__init__()

            number_of_layers = int(len(weights) / 2)
            self._main_layers = []
            for i in range(0, number_of_layers):
                self._main_layers.append(SparseLayer(weights[i * 2], weights[i * 2 + 1], mask[i * 2]))
                # do not add activation on the last layer
                if i != number_of_layers - 1:
                    self._main_layers.append(keras.layers.Activation("relu"))

        def call(self, inputs, **kwargs):
            if type(inputs) is tuple:
                Z = inputs[0]
            else:
                Z = inputs

            for layer in self._main_layers:
                Z = layer(Z)
            return Z

    model = SparseMLP(weights_in, mask_in)
    return model
