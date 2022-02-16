"""

"""
import abc
from src.load import FileGeneric, CNNFile
from tensorflow import keras
from logging import getLogger
logger = getLogger(__name__)

class Model(abc.ABC):
    """

    """
    def __init__(self):
        """

        """
        self.model = None

    @abc.abstractmethod
    def create_model(self, *args) -> None:
        """

        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, test_data: list[FileGeneric], *args) -> list:
        """

        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, filename: str) -> None:
        """

        """
        raise NotImplementedError


class CNNModel(abc.ABC, Model):

    @abc.abstractmethod
    def train(self, train_data: list[FileGeneric], *args) -> None:
        """

        """
        raise NotImplementedError


class ElectrostaticsModel(CNNModel, Model):
    """

    """
    def __init__(self):
        super().__init__()

    def create_model(self, x_dim_size, y_dim_size, z_dim_size):
        logger.info("Creating model")
        model = keras.Sequential()

        model.add(keras.layers.Conv3D(64,
                                      kernel_size=(5, 5, 5),
                                      strides=(1, 1, 1),
                                      activation='relu',
                                      input_shape=(x_dim_size,
                                                   y_dim_size,
                                                   z_dim_size,
                                                   1),
                                      padding="same"))
        model.add(
            keras.layers.Conv3D(64, kernel_size=(5, 5, 5), activation='relu', padding="same"))
        model.add(keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(3, activation='softmax'))

        model.summary()

        model.compile(optimizer=keras.optimizers.SGD(lr=0.0007, momentum=0.9, decay=10 / 40000),
                      loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])
        self.model = model
        logger.info("Created model")

    def train(self, training_files: list[CNNFile], batch_size: int = 256,
              epochs: int = 10, validation_split: float = .2, verbose: int = 2) -> None:
        """

        """
        train_data = [x.data for x in training_files]
        logger.info("Beginning training")
        self.model.fit(x=[x.values for x in train_data],
                       y=[x.label_index for x in training_files],
                       batch_size=batch_size, epochs=epochs,
                       validation_split=validation_split, verbose=verbose)

    def evaluate(self, test_files: list[CNNFile], verbose=2) -> list:
        """

        """
        train_data = [x.data for x in test_files]
        logger.info("Beginning evaluating test set")
        return self.model.evaluate([x.values for x in train_data],
                                   [x.label_index for x in test_files],
                                   verbose=verbose)
