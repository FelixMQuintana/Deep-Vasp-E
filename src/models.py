"""
Module is responsible for holding different types of models that this project can support
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
    Class is an abstract class providing structure on how models should be made. Models must contain a way to
    """
    model: object = dataclasses.field(default=None)

    @abc.abstractmethod
    def create_model(self, *args) -> None:
        """
        Responsible for creating the model of a given class

        :param args: Arguments needed for creating a given model

        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, test_data: [DataGeneric], *args) -> list:
        """

        :param test_data: list of data to be used for evaluation of model.
        :param args: any other arguments that would be needed.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, filename: str) -> None:
        """
        loads

        """
        raise NotImplementedError


class CNNModel(abc.ABC):
    model: Sequential = dataclasses.field(default=None)

    @abc.abstractmethod
    def train(self, train_data: [DataGeneric], *args) -> None:
        """

        """
        raise NotImplementedError


class ElectrostaticsModel(CNNModel, Model):
    """
    Model is responsible for electrostatic voxel data.
    """

    def create_model(self, x_dim_size: int, y_dim_size: int, z_dim_size: int, *args) -> None:
        """
        Create model

        :param x_dim_size: dimension size in x direction.
        :param y_dim_size: dimension size in y direction.
        :param z_dim_size: dimension size in z direction.

        """
        if self.model is not None:
            logger.info("Model has already been created.")
            return None
        self.model = Sequential()
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

    def train(self, training_files: [VoxelData], batch_size: int = 256,
              epochs: int = 10, validation_split: float = .2, verbose: int = 2, *args) -> None:
        """
        Trains the CNN with the given data and allows tuning of how the training is performed.

        :param training_files: list of data to be parsed for training
        :param batch_size: the batch size for training
        :param epochs: number of epochs for training
        :param validation_split: validation split for training
        :param verbose: verbosity of model training

        """
        train_data = [x.values for x in training_files]
        labels = [x.label_index for x in training_files]
        logger.info("Beginning training")
        self.model.fit(x=train_data,
                       y=labels,
                       batch_size=batch_size, epochs=epochs,
                       validation_split=validation_split, verbose=verbose)

    def evaluate(self, test_files: [VoxelData], verbose=2) -> list:
        """
        Method evaluates test data and returns a list of the results

        :param test_files: list of data to be used for testing model
        :param verbose: verbosity of data.

        :return: A list of the resulting predictions( scores array)

        """
        test_data = [x.values for x in test_files]
        labels = [x.label_index for x in test_files]
        logger.info("Beginning evaluating test set")
        return self.model.evaluate(test_data,
                                   labels,
                                   verbose=verbose)

    def load_model(self, filename: Path) -> None:
        """
        Loads a previously trained model

        """
        # TODO: Need to add error handling for a wrong file type fed to this method
        self.model = load_model(filename.stem)
