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

    # input is is (7, 11, 4)
    x = layers.Conv2D(64, 5, activation='elu', kernel_initializer='he_normal', padding='same')(x)
    x = layers.Conv2D(64, 3, activation='elu', kernel_initializer='he_normal', padding='valid')(x)
    x = layers.Conv2D(128, 3, activation='elu', kernel_initializer='he_normal', padding='valid')(x)
    x = layers.Conv2D(128, 3, activation='elu', kernel_initializer='he_normal', padding='valid')(x)

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
