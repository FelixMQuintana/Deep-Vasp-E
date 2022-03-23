"""
Module's purpose is to hold all the information relevant to visualization.
"""
import abc
import dataclasses
from logging import getLogger

import numpy as np
from keras.backend import exp

from src.load import DataGeneric, CNNFile, VoxelData
from src.models import CNNModel
from tensorflow import GradientTape
from tensorflow.keras.models import Model
from pathlib import Path
from tensorflow.keras.models import Sequential

logger = getLogger(__name__)


@dataclasses.dataclass
class VisualizationGeneric(abc.ABC):
    """
    Provide generic structure of what a visualization technique should hold and provide.

    :param visual_data: Data that will be used for visualization.
    """
    visual_data: DataGeneric

    @abc.abstractmethod
    def calculate_visual(self, data: DataGeneric, model: Model) -> None:
        """
        :param data: Data to be visualized
        :param model: model to be used for visualization such as a CNN, Logistic Regression etc.

        :return: None
        """
        raise NotImplemented

    @abc.abstractmethod
    def generate_visual(self, file_path: Path) -> None:
        """
        Method is responsible for generating visual.

        :param file_path: file path to be used for writing visual.

        :return: None
        """
        raise NotImplemented


@dataclasses.dataclass
class NeuralNetwork:
    """
    All neural networks should contain a class index and layer index of interest for visualization

    """
    class_index: int
    layer_index: int


class VoxelGradCam(VisualizationGeneric, NeuralNetwork):
    """
    Class generates gradient based class activation map for voxelized based data(VoxelData).
    """
    visual_data: VoxelData

    def calculate_visual(self, data: VoxelData, model: CNNModel) -> None:
        """
        Calculates gradient class activation map with voxel based data. Follows grad-cam++ with a 3-D adaption.
        https://arxiv.org/abs/1710.11063

        :param data: voxel data to be visualized.
        :param model: An CNNModel to have layers visualized.

        :return:None
        """
        grad_model = Model([model.model.inputs], [model.model.layers[self.layer_index].output, model.model.output])
        image = [data.values]
        with GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, self.class_index]
        outputs = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]
        first = exp(loss) * grads
        second = exp(loss) * grads * grads
        third = exp(loss) * grads * grads * grads
        global_sum = np.sum(outputs, axis=(0, 1, 2))
        alpha_num = second
        alpha_denominator = second * 2 + third * global_sum + 1e-10
        alpha = alpha_num / alpha_denominator
        weights = np.sum(alpha * np.maximum(first, 0.0), axis=(0, 1, 2))
        values = np.zeros(shape=(data.dimensions.x_dim, data.dimensions.y_dim, data.dimensions.z_dim))
        for index, weight in enumerate(weights):
            values += weight * outputs[:, :, :, index]

        self.visual_data = VoxelData(filepath=Path('/DeepVasp-E/VoxelGradCam.CNN'), label_index=data.label_index,
                                     values=values, resolution=data.resolution, x_bounds=data.x_bounds,
                                     y_bounds=data.y_bounds, z_bounds=data.z_bounds, dimensions=data.dimensions)

    def generate_visual(self, file_path: Path) -> None:
        """
        Method is responsible for generating visual.

        :param file_path: file path to be used for writing visual.

        :return: None
        """
        self.visual_data.filepath = Path
        CNNFile.write(data=self.visual_data)

