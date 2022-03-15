"""

"""
import abc
import dataclasses
from pathlib import Path
from src.load import DataGeneric, VoxelData
from logging import getLogger
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import losses

logger = getLogger(__name__)


@dataclasses.dataclass
class Model(abc.ABC):
    """

    """
    model: object

    @abc.abstractmethod
    def create_model(self, *args) -> None:
        """

        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, test_data: list[DataGeneric], *args) -> list:
        """

        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, filename: str) -> None:
        """

        """
        raise NotImplementedError


class CNNModel(Model, abc.ABC):

    @abc.abstractmethod
    def train(self, train_data: list[DataGeneric], *args) -> None:
        """

        """
        raise NotImplementedError


class ElectrostaticsModel(CNNModel, Model):
    """

    """

    def __init__(self):
        self.model: Sequential = Sequential()

    def create_model(self, x_dim_size, y_dim_size, z_dim_size):
        logger.info("Creating model")
        self.model.add(layers.Conv3D(64,
                                     kernel_size=(5, 5, 5),
                                     strides=(1, 1, 1),
                                     activation='relu',
                                     input_shape=(x_dim_size,
                                                  y_dim_size,
                                                  z_dim_size,
                                                  1),
                                     padding="same"))
        self.model.add(
            layers.Conv3D(64, kernel_size=(5, 5, 5), activation='relu', padding="same"))
        self.model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(3, activation='softmax'))

        self.model.summary()

        self.model.compile(optimizer=SGD(lr=0.0007, momentum=0.9, decay=10 / 40000),
                           loss=losses.sparse_categorical_crossentropy,
                           metrics=['accuracy'])
        logger.info("Created model")

    def train(self, training_files: list[VoxelData], batch_size: int = 256,
              epochs: int = 10, validation_split: float = .2, verbose: int = 2) -> None:
        """

        """
        train_data = [x.values for x in training_files]
        labels = [x.label_index for x in training_files]
        logger.info("Beginning training")
        self.model.fit(x=train_data,
                       y=labels,
                       batch_size=batch_size, epochs=epochs,
                       validation_split=validation_split, verbose=verbose)

    def evaluate(self, test_files: list[VoxelData], verbose=2) -> list:
        """

        """
        test_data = [x.values for x in test_files]
        labels = [x.label_index for x in test_files]
        logger.info("Beginning evaluating test set")
        return self.model.evaluate(test_data,
                                   labels,
                                   verbose=verbose)

    def load_model(self, filename: Path) -> None:
        """

        """
        self.model = load_model(filename.stem)
