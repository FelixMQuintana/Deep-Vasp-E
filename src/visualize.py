"""

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

logger = getLogger(__name__)


@dataclasses.dataclass
class VisualizationGeneric(abc.ABC):
    visual_data: DataGeneric

    @abc.abstractmethod
    def calculate_visual(self, data: DataGeneric, model: Model) -> None:
        """

        """
        raise NotImplemented

    @abc.abstractmethod
    def generate_visual(self) -> None:
        """

        """
        raise NotImplemented


@dataclasses.dataclass
class NeuralNetwork:
    class_index: int
    layer_index: int


class VoxelGradCam(VisualizationGeneric, NeuralNetwork):
    """

    """
    visual_data: VoxelData

    def calculate_visual(self, data: VoxelData, model: CNNModel) -> None:
        """

        """
        grad_model = Model([model.inputs], [model.layers[self.layer_index].output, model.output])
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
        for index, w in enumerate(weights):
            values += w * outputs[:, :, :, index]

        self.visual_data = VoxelData(filepath=Path('/DeepVasp-E/VoxelGradCam.CNN'), label_index=data.label_index,
                                     values=values, resolution=data.resolution, x_bounds=data.x_bounds,
                                     y_bounds=data.y_bounds, z_bounds=data.z_bounds, dimensions=data.dimensions)

    def generate_visual(self) -> None:
        """

        """
        CNNFile.write(data=self.visual_data)
