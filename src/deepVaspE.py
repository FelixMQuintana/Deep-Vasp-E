import dataclasses
from pathlib import Path
from enum import Enum
from src.load import CNNFile, VoxelData
from src.models import ElectrostaticsModel


class ProjectStructure(Enum):
    training = "training"
    test = "test"


@dataclasses.dataclass
class Training:
    set: [VoxelData] = dataclasses.field(init=False)


class DeepVaspE:

    def __init__(self, working_directory: Path, evaluation_image: Path = None):
        """

        """
        self._working_directory = working_directory
        self._evaluation_image = evaluation_image
        self._training = Training()
        self._test_set = []
        self._model = ElectrostaticsModel()
        self._training_set_directory = None
        self._test_set_directory = None
        for child in self._working_directory.iterdir():
            if child.name == ProjectStructure.test.value:
                self._test_set_directory = child
            elif child.name == ProjectStructure.training.value:
                self._training_set_directory = child
        if self._training_set_directory is None or self._test_set_directory is None:
            raise RuntimeError("Couldn't find training or test dataset.")

    def load_data_sets(self):
        """

        """
     #   training = []
        for label in self._training_set_directory.iterdir():
            for sample in label.iterdir():
                training.append(CNNFile.load(sample))
        for label in self._training_set_directory.iterdir():
           for sample in label.iterdir():
                self._test_set.append(CNNFile.load(sample))
        self._training.set = training

    def train_model(self):
        self._model.create_model(self._training.set[0])
