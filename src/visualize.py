"""

"""
import abc
from logging import getLogger
from src.load import DataGeneric, FileGeneric, VoxelData
from src.models import Model, CNNModel
logger = getLogger(__name__)


class VisualizationGeneric(abc.ABC):

    @abc.abstractmethod
    def process(self, data: DataGeneric, model: Model) -> None:
        """

        """
        raise NotImplemented

    @abc.abstractmethod
    def generate_visual(self) -> FileGeneric:
        """

        """
        raise NotImplemented


class VoxelGradCam(VisualizationGeneric):

    def process(self, data: VoxelData, model: CNNModel) -> None:
        grad_model = Model([model.inputs], [model.layers[self._conv_layer_index].output, model.output])
        example_image_data, x_dim, y_dim, z_dim, x_bounds, y_bounds, z_bounds = Preprocessing.voxel_parser(
            self._example_image)
        image = np.empty((1, x_dim, y_dim, z_dim, 1))
        image[0] = example_image_data
        with GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, self._class_index]
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
        self._raw_cnn_data = np.zeros(shape=(x_dim, y_dim, z_dim))
        for index, w in enumerate(weights):
            self._raw_cnn_data += w * outputs[:, :, :, index]
        self._create_cnn_file()
