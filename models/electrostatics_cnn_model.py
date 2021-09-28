"""
Module is responsible for creating keras model
"""
import tensorflow as tf
from tensorflow import keras


def create_model(x_dim: int, y_dim: int, z_dim:int):
    """
    Return the keras model given the dimensions of a input protein image
    :param x_dim: x dimension size based on inputs data of model.
    :param y_dim: y dimension size based on inputs data of model.
    :param z_dim: z dimension size based on inputs data of model.
    """
    model = keras.Sequential()

    model.add(keras.layers.Conv3D(64, kernel_size=(5, 5, 5), strides=(1, 1, 1),  # 64
                                  activation='relu',
                                  input_shape=(x_dim, y_dim, z_dim, 1), padding="same"))  # number of channels is 1
    model.add(keras.layers.Conv3D(64, kernel_size=(5, 5, 5), activation='relu', padding="same"))  # 128 #this is gone
    model.add(keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0007, momentum=0.9, decay=10 / 40000),
                  # model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model
