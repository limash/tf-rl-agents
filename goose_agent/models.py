# move all imports inside functions to use ray.remote multitasking

def mlp_layer(x):
    from tensorflow import keras
    import tensorflow.keras.layers as layers

    # initializer = "he_normal"
    initializer = keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_in', distribution='truncated_normal')

    x = layers.Dense(1000, kernel_initializer=initializer,
                     kernel_regularizer=keras.regularizers.l2(0.01),
                     use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    # x = layers.ReLU()(x)

    x = layers.Dense(500, kernel_initializer=initializer,
                     kernel_regularizer=keras.regularizers.l2(0.01),
                     use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    # x = layers.ReLU()(x)

    return x


def conv_layer(x):
    import tensorflow.keras.layers as layers
    from tensorflow import keras

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


def get_dqn(input_shape, n_outputs):
    from tensorflow import keras
    import tensorflow.keras.layers as layers

    # this initialization in the last layer decreases variance in the last layer
    initializer = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)

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
    outputs = layers.Dense(n_outputs, kernel_initializer=initializer)(x)

    model = keras.Model(inputs=[inputs], outputs=[outputs])

    return model
