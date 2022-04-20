"""
This module is responsible for running what was needed for paper:
"""
import dataclasses
from pathlib import Path
from enum import Enum
from src.load import CNNFile, VoxelData
from src.models import ElectrostaticsModel


class ProjectStructure(Enum):
    """
    Project structure split: training/0, training/1, etc..
    """
    TRAINING = "training"
    TEST = "test"


@dataclasses.dataclass
class DataSet:
    """Dataset structure"""
    set: [VoxelData]


class DeepVaspE                                                                                                                                                                                                                                                                                                         :
    """
    Class provides how to gather DeepVaspE results
    """

    def __init__(self, working_directory: Path,
                 evaluation_image: Path = None, model_file: Path = None) -> None:
        """

        :param working_directory: current working directory where data would be held
        :param evaluation_image: image to be used for visualization results
        :param model_file: file to contain pretrained model.

        :return: None
        """
        super().__init__()
       # logging.basicConfig(filename='myapp.log', level=logging.INFO)

     #   self.add_argument()
        if model_file is not None:
            self.load(model_file)
        self._working_directory = working_directory
        self._evaluation_image = evaluation_image
        self._training: DataSet = DataSet([])
        self._test: DataSet = DataSet([])
        self._model = ElectrostaticsModel()
        self._training_set_directory = None
        self._test_set_directory = None
        for child in self._working_directory.iterdir():
            if child.name == ProjectStructure.TEST.value:
                self._test_set_directory = child
            elif child.name == ProjectStructure.TRAINING.value:
                self._training_set_directory = child
        if self._training_set_directory is None or self._test_set_directory is None:
            raise RuntimeError("Couldn't find training or test dataset.")

    def load_data_sets(self) -> None:
        """
        loads data into each respective set.

        :return: None
        """
        for label in self._training_set_directory.iterdir():
            for protein in label.iterdir():
                for sample in protein.iterdir():
                    self._training.set.append(CNNFile.load(sample, int(label.stem)))
        for label in self._test_set_directory.iterdir():
            for protein in label.iterdir():
                for sample in protein.iterdir():
                    self._test.set.append(CNNFile.load(sample, int(label.stem)))
        example: VoxelData = self._training.set[0]
        self._model.create_model(example.dimensions.x_dim, example.dimensions.y_dim, example.dimensions.z_dim)

    def train_model(self) -> None:
        """
        Trains model on training set

        :return: None
        """
        self._model.train(training_files=self._training.set,
                          batch_size=128,
                          epochs=10,
                          validation_split=0.2,
                          verbose=2)

    def evaluate(self) -> None:
        """
        Evaluates model on test set

        :return: None
        """
        self._model.evaluate(test_files=self._test.set, verbose=2)

    def load(self, model_file: Path) -> None:
        """
        loads model

        :return: None
        """
        self._model.load_model(model_file)
